import sys
from types import ModuleType
from warnings import warn


class DeprecatableModule(ModuleType):
    def __init__(self, module):
        ModuleType.__init__(self, module.__name__)
        self.__dict__.update(module.__dict__)

    def __getattribute__(self, name):
        base_getattr = ModuleType.__getattribute__
        try:
            deprecated = base_getattr(self, '_deprecated_members')
        except AttributeError:
            deprecated = {}
            self._deprecated_members = deprecated

        result = base_getattr(self, name)
        warning_text = deprecated.get(name)
        if warning_text is not None:
            warn(warning_text, DeprecationWarning, stacklevel=2)
        return result


def deprecate_module_member(mod_name, name, message):
    current_module = sys.modules[mod_name]
    if not isinstance(current_module, DeprecatableModule):
        current_module = DeprecatableModule(current_module)
        sys.modules[mod_name] = current_module
    current_module._deprecated_members[name] = message
    return None