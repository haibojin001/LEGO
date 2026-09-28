import copy
from collections import deque

import numpy as np

from . import pyll
from .base import miscs_update_idxs_vals

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"


class ExprEvaluator:
    def __init__(self, expr, deepcopy_inputs=False, max_program_len=None, memo_gc=True):
        self.expr = pyll.as_apply(expr)

        if deepcopy_inputs not in (0, 1, False, True):
            raise ValueError("deepcopy_inputs should be bool", deepcopy_inputs)

        self.deepcopy_inputs = deepcopy_inputs
        if max_program_len is None:
            max_program_len = pyll.base.DEFAULT_MAX_PROGRAM_LEN
        self.max_program_len = max_program_len
        self.memo_gc = memo_gc

    def __call__(self, memo=None):
        return self.eval_nodes(memo)[self.expr]

    def eval_nodes(self, memo=None):
        if memo is None:
            memo = {}
        else:
            memo = dict(memo)

        if self.memo_gc:
            clients = {}
            self.clients = clients
            for node in pyll.dfs(self.expr):
                clients.setdefault(node, set())
                for argument in node.inputs():
                    clients.setdefault(argument, set()).add(node)

        pending = deque([self.expr])

        while pending:
            if len(pending) > self.max_program_len:
                raise RuntimeError("Probably infinite loop in document")

            node = pending.pop()

            if node in memo:
                continue

            if node.name == "switch":
                dependencies = self.on_switch(memo, node)
                if dependencies is None:
                    continue
            elif isinstance(node, pyll.Literal):
                self.set_in_memo(memo, node, node.obj)
                continue
            else:
                dependencies = [arg for arg in node.inputs() if arg not in memo]

            if dependencies:
                pending.append(node)
                pending.extend(dependencies)
                continue

            value = self.on_node(memo, node)

            if isinstance(value, pyll.Apply):
                evaluator = self.__class__(
                    value,
                    self.deepcopy_inputs,
                    self.max_program_len,
                    self.memo_gc,
                )
                value = evaluator(memo)

            self.set_in_memo(memo, node, value)

        return memo

    def set_in_memo(self, memo, k, v):
        if not self.memo_gc:
            memo[k] = v
            return

        assert v is not pyll.base.GarbageCollected
        memo[k] = v

        for argument in k.inputs():
            if all(client in memo for client in self.clients[argument]):
                memo[argument] = pyll.base.GarbageCollected

    def on_switch(self, memo, node):
        selector_node = node.pos_args[0]

        if selector_node not in memo:
            return [selector_node]

        selector = memo[selector_node]

        try:
            integer_selector = int(selector)
        except BaseException:
            raise TypeError("switch argument was", selector)

        if selector != integer_selector or selector < 0:
            raise ValueError("switch pos must be positive int", selector)

        selected_node = node.pos_args[integer_selector + 1]

        if selected_node not in memo:
            return [selected_node]

        self.set_in_memo(memo, node, memo[selected_node])
        return None

    def on_node(self, memo, node):
        positional = [memo[arg] for arg in node.pos_args]
        keyword = {name: memo[arg] for name, arg in node.named_args}

        if self.memo_gc:
            for value in positional:
                assert value is not pyll.base.GarbageCollected
            for value in keyword.values():
                assert value is not pyll.base.GarbageCollected

        if self.deepcopy_inputs:
            positional = copy.deepcopy(positional)
            keyword = copy.deepcopy(keyword)

        return pyll.scope._impls[node.name](*positional, **keyword)


class SuggestAlgo(ExprEvaluator):
    """Base evaluator for search algorithms that construct trial suggestions."""

    def __init__(self, domain, trials, seed):
        ExprEvaluator.__init__(self, domain.s_idxs_vals)
        self.domain = domain
        self.trials = trials
        self.rng = np.random.default_rng(seed)

    def __call__(self, new_ids):
        self._trials = self.trials
        self._seed = self.rng.integers(2**31 - 1)
        return self.suggest(new_ids)

    def suggest(self, new_ids):
        new_trials = []

        for new_id in new_ids:
            rng = np.random.default_rng(self._seed + new_id)
            memo = self.eval_nodes(
                memo={
                    self.domain.s_new_ids: [new_id],
                    self.domain.s_rng: rng,
                }
            )
            idxs, vals = memo[self.expr]

            result = self.domain.new_result()
            misc = {
                "tid": new_id,
                "cmd": self.domain.cmd,
                "workdir": self.domain.workdir,
            }
            miscs_update_idxs_vals([misc], idxs, vals)

            new_trials.extend(
                self.trials.new_trial_docs(
                    tids=[new_id],
                    specs=[None],
                    results=[result],
                    miscs=[misc],
                )
            )

        return new_trials

    def on_node(self, memo, node):
        if node.name == "hyperopt_param":
            label = memo[node.pos_args[0]]
            return self.on_node_hyperparameter(memo, node, label)
        return ExprEvaluator.on_node(self, memo, node)

    def on_node_hyperparameter(self, memo, node, label):
        raise NotImplementedError()