from inspect import isclass

from .props import props


class SubclassWithMeta_Meta(type):
    _meta = None

    def __str__(cls):
        meta = cls._meta
        return meta.name if meta else cls.__name__

    def __repr__(cls):
        return "<{} meta={}>".format(cls.__name__, repr(cls._meta))


class SubclassWithMeta(metaclass=SubclassWithMeta_Meta):
    """Base class supporting Meta configuration on subclasses."""

    def __init_subclass__(cls, **meta_options):
        meta = getattr(cls, "Meta", None)
        values = {}

        if meta:
            if isinstance(meta, dict):
                values = meta
            elif isclass(meta):
                values = props(meta)
            else:
                raise Exception(
                    f"Meta have to be either a class or a dict. Received {meta}"
                )
            delattr(cls, "Meta")

        options = dict(meta_options, **values)
        is_abstract = options.pop("abstract", False)

        if is_abstract:
            assert not options, (
                "Abstract types can only contain the abstract attribute. "
                f"Received: abstract, {', '.join(options)}"
            )
            return

        parent = super(cls, cls)
        initializer = getattr(parent, "__init_subclass_with_meta__", None)
        if initializer is not None:
            initializer(**options)

    @classmethod
    def __init_subclass_with_meta__(cls, **meta_options):
        pass