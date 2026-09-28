"""Random-search suggestion utilities."""

import logging

import numpy as np

from . import pyll
from .base import miscs_update_idxs_vals


logger = logging.getLogger(__name__)


def suggest(new_ids, domain, trials, seed):
    rng = np.random.default_rng(seed)
    documents = []

    for new_id in new_ids:
        idxs, vals = pyll.rec_eval(
            domain.s_idxs_vals,
            memo={
                domain.s_new_ids: [new_id],
                domain.s_rng: rng,
            },
        )

        result = domain.new_result()
        misc = {
            "tid": new_id,
            "cmd": domain.cmd,
            "workdir": domain.workdir,
        }
        miscs_update_idxs_vals([misc], idxs, vals)

        documents.extend(
            trials.new_trial_docs(
                [new_id],
                [None],
                [result],
                [misc],
            )
        )

    return documents


def suggest_batch(new_ids, domain, trials, seed):
    rng = np.random.default_rng(seed)
    return pyll.rec_eval(
        domain.s_idxs_vals,
        memo={
            domain.s_new_ids: new_ids,
            domain.s_rng: rng,
        },
    )