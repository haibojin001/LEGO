import inspect
import os
import sys
from contextlib import contextmanager

from mando import Program

try:
    import tomllib
    TOMLLIB_PRESENT = True
except ImportError:
    try:
        import tomli as tomllib
        TOMLLIB_PRESENT = True
    except ImportError:
        TOMLLIB_PRESENT = False

import radon.complexity as cc_mod
from radon.cli.colors import BRIGHT, RED, RESET
from radon.cli.harvest import CCHarvester, HCHarvester, MIHarvester, RawHarvester

if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser


CONFIG_SECTION_NAME = 'radon'


class FileConfig(object):
    """Read defaults for CLI options from the supported config files."""

    def __init__(self):
        self.file_cfg = self.file_config()

    def get_value(self, key, type, default):
        if not self.file_cfg.has_option(CONFIG_SECTION_NAME, key):
            return default
        if type == int:
            return self.file_cfg.getint(
                CONFIG_SECTION_NAME, key, fallback=default
            )
        if type == bool:
            return self.file_cfg.getboolean(
                CONFIG_SECTION_NAME, key, fallback=default
            )
        return self.file_cfg.get(CONFIG_SECTION_NAME, key, fallback=default)

    @staticmethod
    def toml_config():
        if not TOMLLIB_PRESENT:
            return {}

        try:
            with open('pyproject.toml', 'rb') as stream:
                contents = tomllib.load(stream)
            return contents['tool']
        except tomllib.TOMLDecodeError:
            raise
        except Exception:
            return {}

    @staticmethod
    def file_config():
        """Return configuration gathered from the usual config locations."""
        parser = configparser.ConfigParser()

        for filename in (os.getenv('RADONCFG', None), 'radon.cfg'):
            if filename is not None and os.path.exists(filename):
                with open(filename) as stream:
                    parser.read_file(stream)

        parser.read_dict(FileConfig.toml_config())
        parser.read(['setup.cfg', os.path.expanduser('~/.radon.cfg')])
        return parser


class Config(object):
    """A lightweight container used to pass options to harvesters."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        values = ', '.join(
            '{}={!r}'.format(key, value)
            for key, value in sorted(self.__dict__.items())
        )
        return '{}({})'.format(self.__class__.__name__, values)


_cfg = FileConfig()

program = Program(version=sys.modules['radon'].__version__)


def log(message, stream=sys.stdout):
    """Write a single result line, emphasizing error messages."""
    if message.startswith('ERROR'):
        message = '{}{}{}{}'.format(BRIGHT, RED, message, RESET)
    stream.write(message)
    stream.write('\n')


def log_result(harvester, **kwargs):
    """Serialize and write the results produced by a harvester."""
    if kwargs.get('json'):
        formatter = harvester.as_json
    elif kwargs.get('xml'):
        formatter = harvester.as_xml
    elif kwargs.get('md'):
        formatter = harvester.as_md
    elif kwargs.get('codeclimate'):
        formatter = harvester.as_codeclimate_issues
    else:
        formatter = harvester.to_terminal

    stream = kwargs.get('stream', sys.stdout)

    if inspect.isgeneratorfunction(formatter):
        for item in formatter():
            log(item, stream)
    else:
        log(formatter(), stream)


@contextmanager
def outstream(filename):
    """Yield stdout or a text file selected as command output."""
    if filename is None:
        yield sys.stdout
    else:
        with open(filename, 'w') as stream:
            yield stream


@program.command
@program.arg('paths', nargs='+')
def cc(
    paths,
    min=_cfg.get_value('cc_min', str, 'A'),
    max=_cfg.get_value('cc_max', str, 'F'),
    show_complexity=_cfg.get_value('show_complexity', bool, False),
    average=_cfg.get_value('average', bool, False),
    exclude=_cfg.get_value('exclude', str, None),
    ignore=_cfg.get_value('ignore', str, None),
    order=_cfg.get_value('order', str, 'SCORE'),
    json=False,
    no_assert=_cfg.get_value('no_assert', bool, False),
    show_closures=_cfg.get_value('show_closures', bool, False),
    total_average=_cfg.get_value('total_average', bool, False),
    xml=False,
    md=False,
    codeclimate=False,
    output_file=_cfg.get_value('output_file', str, None),
    include_ipynb=_cfg.get_value('include_ipynb', bool, False),
    ipynb_cells=_cfg.get_value('ipynb_cells', bool, False),
):
    """Analyze Python sources and calculate cyclomatic complexity."""
    config = Config(
        min=min.upper(),
        max=max.upper(),
        exclude=exclude,
        ignore=ignore,
        show_complexity=show_complexity,
        average=average,
        total_average=total_average,
        order=getattr(cc_mod, order.upper(), getattr(cc_mod, 'SCORE')),
        no_assert=no_assert,
        show_closures=show_closures,
        include_ipynb=include_ipynb,
        ipynb_cells=ipynb_cells,
    )
    harvester = CCHarvester(paths, config)
    with outstream(output_file) as stream:
        log_result(
            harvester,
            json=json,
            xml=xml,
            md=md,
            codeclimate=codeclimate,
            stream=stream,
        )


@program.command
@program.arg('paths', nargs='+')
def raw(
    paths,
    exclude=_cfg.get_value('exclude', str, None),
    ignore=_cfg.get_value('ignore', str, None),
    summary=False,
    json=False,
    output_file=_cfg.get_value('output_file', str, None),
    include_ipynb=_cfg.get_value('include_ipynb', bool, False),
    ipynb_cells=_cfg.get_value('ipynb_cells', bool, False),
):
    """Analyze Python sources and report raw code metrics."""
    config = Config(
        exclude=exclude,
        ignore=ignore,
        summary=summary,
        include_ipynb=include_ipynb,
        ipynb_cells=ipynb_cells,
    )
    harvester = RawHarvester(paths, config)
    with outstream(output_file) as stream:
        log_result(harvester, json=json, stream=stream)


@program.command
@program.arg('paths', nargs='+')
def mi(
    paths,
    min=_cfg.get_value('mi_min', str, 'A'),
    max=_cfg.get_value('mi_max', str, 'C'),
    multi=_cfg.get_value('multi', bool, True),
    exclude=_cfg.get_value('exclude', str, None),
    ignore=_cfg.get_value('ignore', str, None),
    show=_cfg.get_value('show_mi', bool, False),
    json=False,
    sort=False,
    output_file=_cfg.get_value('output_file', str, None),
    include_ipynb=_cfg.get_value('include_ipynb', bool, False),
    ipynb_cells=_cfg.get_value('ipynb_cells', bool, False),
):
    """Analyze Python sources and calculate maintainability indexes."""
    config = Config(
        min=min.upper(),
        max=max.upper(),
        multi=multi,
        exclude=exclude,
        ignore=ignore,
        show=show,
        sort=sort,
        include_ipynb=include_ipynb,
        ipynb_cells=ipynb_cells,
    )
    harvester = MIHarvester(paths, config)
    with outstream(output_file) as stream:
        log_result(harvester, json=json, stream=stream)


@program.command
@program.arg('paths', nargs='+')
def hal(
    paths,
    exclude=_cfg.get_value('exclude', str, None),
    ignore=_cfg.get_value('ignore', str, None),
    json=False,
    output_file=_cfg.get_value('output_file', str, None),
    include_ipynb=_cfg.get_value('include_ipynb', bool, False),
    ipynb_cells=_cfg.get_value('ipynb_cells', bool, False),
):
    """Analyze Python sources and calculate Halstead metrics."""
    config = Config(
        exclude=exclude,
        ignore=ignore,
        include_ipynb=include_ipynb,
        ipynb_cells=ipynb_cells,
    )
    harvester = HCHarvester(paths, config)
    with outstream(output_file) as stream:
        log_result(harvester, json=json, stream=stream)


def main(args=None):
    """Run the radon command-line application."""
    if args is None:
        args = sys.argv[1:]
    return program.execute(args)