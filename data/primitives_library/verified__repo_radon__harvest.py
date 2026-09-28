"""Harvesting helpers used by the command-line interface."""

import collections
import json
import sys
from builtins import super

from radon.cli.colors import MI_RANKS, RANKS_COLORS, RESET
from radon.cli.tools import (
    _open,
    cc_to_dict,
    cc_to_terminal,
    dict_to_codeclimate_issues,
    dict_to_md,
    dict_to_xml,
    iter_filenames,
    raw_to_dict,
    strip_ipython,
)
from radon.complexity import add_inner_blocks, cc_rank, cc_visit, sorted_results
from radon.metrics import h_visit, mi_rank, mi_visit
from radon.raw import analyze

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

try:
    import nbformat

    SUPPORTS_IPYNB = True
except ImportError:
    SUPPORTS_IPYNB = False


class Harvester(object):
    """Common implementation shared by all metric harvesters."""

    def __init__(self, paths, config):
        self.paths = paths
        self.config = config
        self._results = []

    def _iter_filenames(self):
        return iter_filenames(
            self.paths,
            self.config.exclude,
            self.config.ignore,
        )

    def gobble(self, fobj):
        raise NotImplementedError

    def run(self):
        for filename in self._iter_filenames():
            with _open(filename) as fobj:
                try:
                    if filename.endswith(".ipynb"):
                        if SUPPORTS_IPYNB and self.config.include_ipynb:
                            notebook = nbformat.read(
                                fobj,
                                as_version=nbformat.NO_CONVERT,
                            )
                            sources = [
                                cell.source
                                for cell in notebook.cells
                                if cell.cell_type == "code"
                            ]
                            document = "\n".join(sources)
                            yield (
                                filename,
                                self.gobble(StringIO(strip_ipython(document))),
                            )

                            if self.config.ipynb_cells:
                                for index, source in enumerate(sources):
                                    yield (
                                        "{}:[{}]".format(filename, index),
                                        self.gobble(
                                            StringIO(strip_ipython(source))
                                        ),
                                    )
                    else:
                        yield filename, self.gobble(fobj)
                except Exception as exc:
                    yield filename, {"error": str(exc)}

    @property
    def results(self):
        def cached(iterator, cache):
            for item in iterator:
                yield item
                cache.append(item)

        if self._results:
            return self._results
        return cached(self.run(), self._results)

    def as_json(self):
        return json.dumps(dict(self.results))

    def as_xml(self):
        raise NotImplementedError

    def as_md(self):
        raise NotImplementedError

    def as_codeclimate_issues(self):
        raise NotImplementedError

    def to_terminal(self):
        raise NotImplementedError


class CCHarvester(Harvester):
    """Harvester for cyclomatic-complexity results."""

    def gobble(self, fobj):
        blocks = cc_visit(fobj.read(), no_assert=self.config.no_assert)
        if self.config.show_closures:
            blocks = add_inner_blocks(blocks)
        return sorted_results(blocks, order=self.config.order)

    def _to_dicts(self):
        output = {}
        for filename, blocks in self.results:
            if "error" in blocks:
                output[filename] = blocks
                continue

            values = [
                value
                for value in map(cc_to_dict, blocks)
                if self.config.min <= value["rank"] <= self.config.max
            ]
            if values:
                output[filename] = values
        return output

    def as_json(self):
        return json.dumps(self._to_dicts())

    def as_xml(self):
        return dict_to_xml(self._to_dicts())

    def as_md(self):
        return dict_to_md(self._to_dicts())

    def as_codeclimate_issues(self):
        return dict_to_codeclimate_issues(self._to_dicts(), self.config.min)

    def to_terminal(self):
        total_complexity = 0.0
        blocks_analyzed = 0

        for filename, blocks in self.results:
            if "error" in blocks:
                yield filename, (blocks["error"],), {"error": True}
                continue

            rendered, complexity, number = cc_to_terminal(
                blocks,
                self.config.show_complexity,
                self.config.min,
                self.config.max,
                self.config.total_average,
            )
            total_complexity += complexity
            blocks_analyzed += number

            if rendered:
                yield filename, (), {}
                yield rendered, (), {"indent": 1}

        if (
            (self.config.average or self.config.total_average)
            and blocks_analyzed
        ):
            average = total_complexity / blocks_analyzed
            rank = cc_rank(average)
            yield (
                "\n{0} blocks (classes, functions, methods) analyzed.",
                (blocks_analyzed,),
                {},
            )
            yield (
                "Average complexity: {0}{1} ({2}){3}",
                (RANKS_COLORS[rank], rank, average, RESET),
                {},
            )


class RawHarvester(Harvester):
    """Harvester for raw source-code metrics."""

    headers = [
        "LOC",
        "LLOC",
        "SLOC",
        "Comments",
        "Single comments",
        "Multi",
        "Blank",
    ]

    def gobble(self, fobj):
        return raw_to_dict(analyze(fobj.read()))

    def as_xml(self):
        raise NotImplementedError("RawHarvester: cannot export results as XML")

    def to_terminal(self):
        for filename, result in self.results:
            if "error" in result:
                yield filename, (result["error"],), {"error": True}
                continue

            yield filename, (), {}
            for header in self.headers:
                key = header.lower().replace(" ", "_")
                yield "{0}: {1}", (header, result[key]), {"indent": 1}


class MIHarvester(Harvester):
    """Harvester for maintainability-index results."""

    def gobble(self, fobj):
        return mi_visit(fobj.read(), self.config.multi)

    def _to_dicts(self):
        output = {}
        for filename, value in self.results:
            if isinstance(value, dict) and "error" in value:
                output[filename] = value
                continue

            rank = mi_rank(value)
            if self.config.min <= rank <= self.config.max:
                output[filename] = {"mi": value, "rank": rank}
        return output

    def as_json(self):
        return json.dumps(self._to_dicts())

    def as_xml(self):
        raise NotImplementedError("MIHarvester: cannot export results as XML")

    def to_terminal(self):
        for filename, value in self.results:
            if isinstance(value, dict) and "error" in value:
                yield filename, (value["error"],), {"error": True}
                continue

            rank = mi_rank(value)
            if self.config.min <= rank <= self.config.max:
                if self.config.show:
                    yield (
                        "{0} - {1} ({2:.2f})",
                        (filename, rank, value),
                        {"color": MI_RANKS[rank]},
                    )
                else:
                    yield (
                        "{0} - {1}",
                        (filename, rank),
                        {"color": MI_RANKS[rank]},
                    )


class HHarvester(Harvester):
    """Harvester for Halstead metrics."""

    def gobble(self, fobj):
        return h_visit(fobj.read())

    @staticmethod
    def _report_to_dict(report):
        if hasattr(report, "_asdict"):
            return dict(report._asdict())
        return report

    def _to_dicts(self):
        output = {}
        for filename, result in self.results:
            if isinstance(result, dict) and "error" in result:
                output[filename] = result
                continue

            total = self._report_to_dict(result.total)
            functions = collections.OrderedDict(
                (
                    name,
                    self._report_to_dict(report),
                )
                for name, report in result.functions
            )
            output[filename] = {
                "total": total,
                "functions": functions,
            }
        return output

    def as_json(self):
        return json.dumps(self._to_dicts())

    def as_xml(self):
        raise NotImplementedError("HHarvester: cannot export results as XML")

    def to_terminal(self):
        for filename, result in self.results:
            if isinstance(result, dict) and "error" in result:
                yield filename, (result["error"],), {"error": True}
                continue

            yield filename, (), {}
            values = self._report_to_dict(result.total)
            for key, value in values.items():
                yield "{0}: {1}", (key, value), {"indent": 1}