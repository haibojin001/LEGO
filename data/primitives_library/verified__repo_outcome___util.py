from typing import Any, Dict


class AlreadyUsedError(RuntimeError):
    """Raised when an Outcome is unwrapped more than once."""


def fixup_module_metadata(
    module_name: str,
    namespace: Dict[str, object],
) -> None:
    def update_metadata(value: object) -> None:
        defining_module = getattr(value, "__module__", None)
        if defining_module is None or not defining_module.startswith("outcome."):
            return

        value.__module__ = module_name

        if isinstance(value, type):
            for member in value.__dict__.values():
                update_metadata(member)

    exported = namespace["__all__"]
    assert isinstance(exported, (list, tuple)), repr(exported)

    for name in exported:
        update_metadata(namespace[name])


def remove_tb_frames(exc: BaseException, n: int) -> BaseException:
    traceback = exc.__traceback__

    for _ in range(n):
        assert traceback is not None
        traceback = traceback.tb_next

    return exc.with_traceback(traceback)