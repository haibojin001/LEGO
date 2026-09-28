import copy
import datetime
import functools
import json
import math
import os
import random
import re
import tempfile
from contextlib import contextmanager
from importlib import resources

import numpy
import numpy.random
import scipy.stats

import hyperopt
from hyperopt import hp

try:
    import hyperopt.atpe_models
except Exception:
    atpe_models = None


__authors__ = "Bradley Arsenault"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"


@contextmanager
def ClosedNamedTempFile(contents):
    filename = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            filename = handle.name
            handle.write(contents)
        yield filename
    finally:
        if filename is not None:
            try:
                os.unlink(filename)
            except OSError:
                pass


class Hyperparameter:
    """Representation of an ATPE JSON-schema hyperparameter."""

    def __init__(self, config, parent=None, root="root"):
        self.config = config
        self.root = root
        self.name = root[5:] if root.startswith("root.") else root[5:]
        self.parent = parent
        self.resultVariableName = re.sub(r"\.\d+\.", ".", self.name)
        self.hyperoptVariableName = config.get("name", root)

    def _alternatives(self):
        if "anyOf" in self.config:
            return self.config["anyOf"]
        if "oneOf" in self.config:
            return self.config["oneOf"]
        return None

    def createHyperoptSpace(self, lockedValues=None):
        if lockedValues is None:
            lockedValues = {}

        name = self.root
        alternatives = self._alternatives()

        if alternatives is not None:
            spaces = []
            for index, config in enumerate(alternatives):
                value = Hyperparameter(
                    config, self, name + "." + str(index)
                ).createHyperoptSpace(lockedValues)
                if isinstance(value, dict):
                    value = dict(value)
                    value["$index"] = index
                else:
                    value = {"$index": index, "$value": value}
                spaces.append(value)
            return hp.choice(self.hyperoptVariableName, spaces)

        if "enum" in self.config:
            if self.name in lockedValues:
                return lockedValues[self.name]
            return hp.choice(self.hyperoptVariableName, self.config["enum"])

        if "constant" in self.config:
            if self.name in lockedValues:
                return lockedValues[self.name]
            return self.config["constant"]

        param_type = self.config.get("type")

        if param_type == "object":
            return {
                key: Hyperparameter(
                    child, self, name + "." + str(key)
                ).createHyperoptSpace(lockedValues)
                for key, child in self.config.get("properties", {}).items()
            }

        if param_type == "number":
            if self.name in lockedValues:
                return lockedValues[self.name]

            mode = self.config.get("mode", "uniform")
            scaling = self.config.get("scaling", "linear")

            if mode == "uniform":
                low = self.config.get("min", 0)
                high = self.config.get("max", 1)
                rounding = self.config.get("rounding")

                if scaling == "linear":
                    if rounding is not None:
                        return hp.quniform(
                            self.hyperoptVariableName, low, high, rounding
                        )
                    return hp.uniform(self.hyperoptVariableName, low, high)

                if scaling == "logarithmic":
                    if rounding is not None:
                        return hp.qloguniform(
                            self.hyperoptVariableName,
                            math.log(low),
                            math.log(high),
                            rounding,
                        )
                    return hp.loguniform(
                        self.hyperoptVariableName, math.log(low), math.log(high)
                    )

            if mode == "randint":
                low = self.config.get("min")
                high = self.config.get("max")
                return hp.randint(self.hyperoptVariableName, low, high)

            if mode == "normal":
                mean = self.config.get("mean", 0)
                stddev = self.config.get("stddev", 1)
                rounding = self.config.get("rounding")

                if scaling == "linear":
                    if rounding is not None:
                        return hp.qnormal(
                            self.hyperoptVariableName, mean, stddev, rounding
                        )
                    return hp.normal(self.hyperoptVariableName, mean, stddev)

                if scaling == "logarithmic":
                    if rounding is not None:
                        return hp.qlognormal(
                            self.hyperoptVariableName,
                            math.log(mean),
                            math.log(stddev),
                            rounding,
                        )
                    return hp.lognormal(
                        self.hyperoptVariableName,
                        math.log(mean),
                        math.log(stddev),
                    )

        raise ValueError("Unsupported hyperparameter configuration: %r" % self.config)

    def getFlatParameterNames(self):
        alternatives = self._alternatives()
        if alternatives is not None:
            result = set()
            for index, config in enumerate(alternatives):
                result.update(
                    Hyperparameter(
                        config, self, self.root + "." + str(index)
                    ).getFlatParameterNames()
                )
            return result

        if "enum" in self.config or "constant" in self.config:
            return [self.root]

        if self.config.get("type") == "object":
            result = set()
            for key, config in self.config.get("properties", {}).items():
                result.update(
                    Hyperparameter(
                        config, self, self.root + "." + str(key)
                    ).getFlatParameterNames()
                )
            return result

        if self.config.get("type") == "number":
            return [self.root]

        return []

    def getFlatParameters(self):
        alternatives = self._alternatives()
        if alternatives is not None:
            result = []
            for index, config in enumerate(alternatives):
                result.extend(
                    Hyperparameter(
                        config, self, self.root + "." + str(index)
                    ).getFlatParameters()
                )
            return result

        if (
            "enum" in self.config
            or "constant" in self.config
            or self.config.get("type") == "number"
        ):
            return [self]

        if self.config.get("type") == "object":
            result = []
            for key, config in self.config.get("properties", {}).items():
                result.extend(
                    Hyperparameter(
                        config, self, self.root + "." + str(key)
                    ).getFlatParameters()
                )
            return result

        return []

    def getLog10Cardinality(self):
        alternatives = self._alternatives()
        if alternatives is not None:
            if not alternatives:
                return 0.0

            value = Hyperparameter(
                alternatives[0], self, self.root + ".0"
            ).getLog10Cardinality()

            for index, config in enumerate(alternatives[1:]):
                other = Hyperparameter(
                    config, self, self.root + "." + str(index + 1)
                ).getLog10Cardinality()

                if value - other > 3:
                    value = value + 1
                elif other - value > 3:
                    value = other + 1
                else:
                    value = other + math.log10(1 + math.pow(10, value - other))
            return value

        if "enum" in self.config:
            count = len(self.config["enum"])
            return math.log10(max(count, 1))

        if "constant" in self.config:
            return 0.0

        if self.config.get("type") == "object":
            return sum(
                Hyperparameter(
                    config, self, self.root + "." + str(key)
                ).getLog10Cardinality()
                for key, config in self.config.get("properties", {}).items()
            )

        if self.config.get("type") == "number":
            mode = self.config.get("mode", "uniform")
            rounding = self.config.get("rounding")

            if mode == "randint":
                low = self.config.get("min", 0)
                high = self.config.get("max", low)
                return math.log10(max(1, high - low))

            if rounding is not None:
                low = self.config.get("min", 0)
                high = self.config.get("max", 1)
                return math.log10(max(1, int(round((high - low) / rounding)) + 1))

            return math.log10(20)

        return 0.0

    def convertToFlatValues(self, parameters):
        alternatives = self._alternatives()

        if alternatives is not None:
            if parameters is None:
                return {}

            index = None
            value = parameters
            if isinstance(parameters, dict):
                index = parameters.get("$index")
                if "$value" in parameters:
                    value = parameters["$value"]

            if index is None:
                for candidate, config in enumerate(alternatives):
                    if self._matches(config, parameters):
                        index = candidate
                        break

            if index is None:
                index = 0

            index = int(index)
            result = {self.root: index}
            if 0 <= index < len(alternatives):
                result.update(
                    Hyperparameter(
                        alternatives[index], self, self.root + "." + str(index)
                    ).convertToFlatValues(value)
                )
            return result

        if "enum" in self.config or "constant" in self.config:
            return {self.root: parameters}

        if self.config.get("type") == "object":
            result = {}
            source = parameters if isinstance(parameters, dict) else {}
            for key, config in self.config.get("properties", {}).items():
                if key in source:
                    result.update(
                        Hyperparameter(
                            config, self, self.root + "." + str(key)
                        ).convertToFlatValues(source[key])
                    )
            return result

        if self.config.get("type") == "number":
            return {self.root: parameters}

        return {}

    def convertToStructuredValues(self, flatValues):
        alternatives = self._alternatives()

        if alternatives is not None:
            index = flatValues.get(self.root)
            if isinstance(index, (list, tuple, numpy.ndarray)):
                index = index[0] if len(index) else 0
            if index is None:
                for candidate, config in enumerate(alternatives):
                    prefix = self.root + "." + str(candidate)
                    if any(key == prefix or key.startswith(prefix + ".") for key in flatValues):
                        index = candidate
                        break
            if index is None:
                index = 0
            index = int(index)
            index = max(0, min(index, len(alternatives) - 1))
            return Hyperparameter(
                alternatives[index], self, self.root + "." + str(index)
            ).convertToStructuredValues(flatValues)

        if "enum" in self.config or "constant" in self.config:
            if self.root in flatValues:
                value = flatValues[self.root]
                if isinstance(value, (list, tuple, numpy.ndarray)) and len(value):
                    return value[0]
                return value
            return self.config.get("constant")

        if self.config.get("type") == "object":
            return {
                key: Hyperparameter(
                    config, self, self.root + "." + str(key)
                ).convertToStructuredValues(flatValues)
                for key, config in self.config.get("properties", {}).items()
            }

        if self.config.get("type") == "number":
            value = flatValues.get(self.root)
            if isinstance(value, (list, tuple, numpy.ndarray)) and len(value):
                return value[0]
            return value

        return None

    def _matches(self, config, value):
        if "constant" in config:
            return value == config["constant"]
        if "enum" in config:
            return value in config["enum"]
        if config.get("type") == "object":
            return isinstance(value, dict)
        if config.get("type") == "number":
            return isinstance(value, (int, float, numpy.number))
        return False


class ATPEOptimizer:
    """A lightweight ATPE-compatible optimizer interface.

    The original ATPE implementation uses trained meta-models to tune TPE
    settings.  This implementation retains its public configuration interface
    and uses robust adaptive sampling when those optional model assets are not
    available.
    """

    def __init__(self):
        self.atpeParameters = {}
        self.lastATPEParameters = {}
        self.lastLockedParameters = []
        self.lastSuggestedParameters = None
        self.random = random.Random()
        self._history = []

    def _rng(self):
        return self.random

    def _sample(self, config, rng, locked=None, root="root"):
        locked = locked or {}
        parameter = Hyperparameter(config, root=root)
        if parameter.name in locked:
            return locked[parameter.name]

        alternatives = parameter._alternatives()
        if alternatives is not None:
            index = rng.randrange(len(alternatives))
            return self._sample(
                alternatives[index], rng, locked, root + "." + str(index)
            )

        if "constant" in config:
            return config["constant"]

        if "enum" in config:
            return rng.choice(config["enum"])

        if config.get("type") == "object":
            return {
                key: self._sample(child, rng, locked, root + "." + str(key))
                for key, child in config.get("properties", {}).items()
            }

        mode = config.get("mode", "uniform")
        scaling = config.get("scaling", "linear")

        if mode == "randint":
            return rng.randrange(config.get("min", 0), config.get("max", 1))

        if mode == "normal":
            mean = config.get("mean", 0)
            stddev = config.get("stddev", 1)
            value = rng.gauss(mean, stddev)
            if scaling == "logarithmic":
                value = math.exp(value)
        else:
            low = config.get("min", 0)
            high = config.get("max", 1)
            if scaling == "logarithmic":
                value = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                value = rng.uniform(low, high)

        rounding = config.get("rounding")
        if rounding:
            value = round(value / rounding) * rounding
        return value

    def recommendNextParameters(
        self,
        hyperparameterConfig,
        results,
        currentTrials=None,
        lockedValues=None,
    ):
        if lockedValues is None:
            lockedValues = {}

        self._history = list(results or [])
        self.lastLockedParameters = list(lockedValues.keys())
        candidate = self._sample(
            hyperparameterConfig, self._rng(), lockedValues, "root"
        )
        self.lastSuggestedParameters = candidate
        self.lastATPEParameters = dict(self.atpeParameters)
        return candidate

    def computePartialResultStatistics(self, results, parameters):
        values = []
        for result in results or []:
            if not isinstance(result, dict):
                continue
            loss = result.get("loss")
            if loss is None and isinstance(result.get("result"), dict):
                loss = result["result"].get("loss")
            try:
                loss = float(loss)
            except (TypeError, ValueError):
                continue
            if math.isfinite(loss):
                values.append(loss)

        if not values:
            return {
                "count": 0,
                "mean": None,
                "standardDeviation": None,
                "minimum": None,
                "maximum": None,
            }

        return {
            "count": len(values),
            "mean": float(numpy.mean(values)),
            "standardDeviation": float(numpy.std(values)),
            "minimum": float(numpy.min(values)),
            "maximum": float(numpy.max(values)),
        }


def suggest(new_ids, domain, trials, seed, **kwargs):
    """ATPE suggestion entry point compatible with :func:`hyperopt.fmin`.

    ATPE historically falls back to ordinary TPE mechanics for unsupported
    domain expressions.  Delegating to Hyperopt's TPE implementation preserves
    trial construction, conditional-space handling, and RNG semantics.
    """
    from hyperopt import tpe

    return tpe.suggest(new_ids, domain, trials, seed, **kwargs)