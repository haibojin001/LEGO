import contextlib

from tqdm import tqdm

from .std_out_err_redirect_tqdm import std_out_err_redirect_tqdm


@contextlib.contextmanager
def tqdm_progress_callback(initial, total):
    with std_out_err_redirect_tqdm() as output:
        with tqdm(
            total=total,
            file=output,
            postfix={"best loss": "?"},
            disable=False,
            dynamic_ncols=True,
            unit="trial",
            initial=initial,
        ) as progress:
            yield progress


@contextlib.contextmanager
def no_progress_callback(initial, total):
    class NoProgressContext:
        def update(self, n):
            return None

    yield NoProgressContext()


default_callback = tqdm_progress_callback
"""Use tqdm for progress by default"""