import logging

import numpy as np

from .algobase import ExprEvaluator, SuggestAlgo
from .base import miscs_to_idxs_vals
from .pyll.stochastic import (
    categorical,
    lognormal,
    loguniform,
    normal,
    qlognormal,
    qloguniform,
    qnormal,
    quniform,
    uniform,
)

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

logger = logging.getLogger(__name__)


class AnnealingAlgo(SuggestAlgo):
    """
    Annealing-based suggestion algorithm.

    The algorithm samples from the prior initially, then increasingly samples
    near values observed in good previous trials.
    """

    def __init__(self, domain, trials, seed, avg_best_idx=2.0, shrink_coef=0.1):
        SuggestAlgo.__init__(self, domain, trials, seed=seed)

        self.avg_best_idx = avg_best_idx
        self.shrink_coef = shrink_coef

        docs_and_losses = {}
        for doc in trials.trials:
            tid = doc["tid"]
            loss = domain.loss(doc["result"], doc["spec"])
            docs_and_losses[tid] = (doc, float("inf") if loss is None else loss)

        self.tid_docs_losses = sorted(docs_and_losses.items())
        self.tids = np.asarray(
            [tid for tid, (_doc, _loss) in self.tid_docs_losses]
        )
        self.losses = np.asarray(
            [loss for _tid, (_doc, loss) in self.tid_docs_losses]
        )
        self.tid_losses_dct = dict(zip(self.tids, self.losses))

        self.node_tids, self.node_vals = miscs_to_idxs_vals(
            [doc["misc"] for _tid, (doc, _loss) in self.tid_docs_losses],
            keys=list(domain.params.keys()),
        )
        self.best_tids = []

    def shrinking(self, label):
        return 1.0 / (1.0 + len(self.node_vals[label]) * self.shrink_coef)

    def choose_ltv(self, label, size):
        """Return loss, tid, and value from a good prior trial."""
        tids = self.node_tids[label]
        vals = self.node_vals[label]
        losses = [self.tid_losses_dct[tid] for tid in tids]

        if size == 1:
            available = set(tids)
            for tid in self.best_tids:
                if tid in available:
                    position = tids.index(tid)
                    return losses[position], tid, vals[position]

        ranks = self.rng.geometric(1.0 / self.avg_best_idx, size=size) - 1
        ranks = np.clip(ranks, 0, len(tids) - 1).astype("int32")

        order = np.argsort(losses)
        picks = order[ranks]

        picked_losses = np.asarray(losses)[picks]
        picked_tids = np.asarray(tids)[picks]
        picked_vals = np.asarray(vals)[picks]

        if size == 1:
            self.best_tids.append(int(picked_tids[0]))

        return picked_losses, picked_tids, picked_vals

    def on_node_hyperparameter(self, memo, node, label):
        if len(self.node_vals[label]) > 0:
            size = memo[node.arg["size"]]
            _loss, tid, val = self.choose_ltv(label, size=size)
            try:
                handler = getattr(self, "hp_%s" % node.name)
            except AttributeError:
                raise NotImplementedError("Annealing", node.name)
            return handler(memo, node, label, tid, val)

        return ExprEvaluator.on_node(self, memo, node)

    def hp_uniform(
        self,
        memo,
        node,
        label,
        tid,
        val,
        log_scale=False,
        pass_q=False,
        uniform_like=uniform,
    ):
        low = memo[node.pos_args[0]]
        high = memo[node.pos_args[1]]
        size = memo[node.arg["size"]]

        if log_scale:
            low = np.log(low)
            high = np.log(high)
            val = np.log(val)

        width = (high - low) * self.shrinking(label)
        center = np.clip(val, low + width / 2.0, high - width / 2.0)

        kwargs = dict(
            low=center - width / 2.0,
            high=center + width / 2.0,
            rng=self.rng,
            size=size,
        )
        if pass_q:
            kwargs["q"] = memo[node.pos_args[2]]

        samples = uniform_like(**kwargs)

        if log_scale and not pass_q:
            samples = np.exp(samples)

        return samples

    def hp_quniform(self, memo, node, label, tid, val):
        return self.hp_uniform(
            memo,
            node,
            label,
            tid,
            val,
            pass_q=True,
            uniform_like=quniform,
        )

    def hp_loguniform(self, memo, node, label, tid, val):
        return self.hp_uniform(
            memo,
            node,
            label,
            tid,
            val,
            log_scale=True,
            uniform_like=loguniform,
        )

    def hp_qloguniform(self, memo, node, label, tid, val):
        return self.hp_uniform(
            memo,
            node,
            label,
            tid,
            val,
            log_scale=True,
            pass_q=True,
            uniform_like=qloguniform,
        )

    def hp_normal(
        self,
        memo,
        node,
        label,
        tid,
        val,
        log_scale=False,
        pass_q=False,
        normal_like=normal,
    ):
        sigma = memo[node.pos_args[1]]
        size = memo[node.arg["size"]]

        if log_scale:
            val = np.log(val)

        sigma = sigma * self.shrinking(label)

        kwargs = dict(mu=val, sigma=sigma, rng=self.rng, size=size)
        if pass_q:
            kwargs["q"] = memo[node.pos_args[2]]

        samples = normal_like(**kwargs)

        if log_scale and not pass_q:
            samples = np.exp(samples)

        return samples

    def hp_qnormal(self, memo, node, label, tid, val):
        return self.hp_normal(
            memo,
            node,
            label,
            tid,
            val,
            pass_q=True,
            normal_like=qnormal,
        )

    def hp_lognormal(self, memo, node, label, tid, val):
        return self.hp_normal(
            memo,
            node,
            label,
            tid,
            val,
            log_scale=True,
            normal_like=lognormal,
        )

    def hp_qlognormal(self, memo, node, label, tid, val):
        return self.hp_normal(
            memo,
            node,
            label,
            tid,
            val,
            log_scale=True,
            pass_q=True,
            normal_like=qlognormal,
        )

    def hp_categorical(self, memo, node, label, tid, val):
        size = memo[node.arg["size"]]
        prior = np.asarray(memo[node.pos_args[0]], dtype="float64")

        chosen = np.asarray(val).reshape(-1)
        counts = np.bincount(chosen.astype("int64"), minlength=len(prior))
        counts = np.asarray(counts, dtype="float64")

        if counts.sum():
            counts /= counts.sum()

        amount = self.shrinking(label)
        p = amount * prior + (1.0 - amount) * counts
        return categorical(p=p, rng=self.rng, size=size)

    def hp_randint(self, memo, node, label, tid, val):
        size = memo[node.arg["size"]]
        upper = int(memo[node.pos_args[0]])
        prior = np.ones(upper, dtype="float64") / upper

        chosen = np.asarray(val).reshape(-1)
        counts = np.bincount(chosen.astype("int64"), minlength=upper)
        counts = np.asarray(counts, dtype="float64")

        if counts.sum():
            counts /= counts.sum()

        amount = self.shrinking(label)
        p = amount * prior + (1.0 - amount) * counts
        return categorical(p=p, rng=self.rng, size=size)


def suggest(new_ids, domain, trials, seed, **kwargs):
    return AnnealingAlgo(domain, trials, seed, **kwargs).batch(new_ids)