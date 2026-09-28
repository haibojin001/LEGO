import sys

import numpy as np

from .pyll import Apply, as_apply, dfs, scope, stochastic, toposort


stoch = stochastic.implicit_stochastic_symbols


def ERR(msg):
    print("hyperopt.vectorize.ERR", msg, file=sys.stderr)


@scope.define_pure
def vchoice_split(idxs, choices, n_options):
    groups = [[] for _ in range(n_options)]
    if len(idxs) != len(choices):
        raise ValueError(
            "idxs and choices different len",
            (len(idxs), len(choices)),
        )
    for idx, choice in zip(idxs, choices):
        groups[choice].append(idx)
    return groups


@scope.define_pure
def vchoice_merge(idxs, choices, *vals):
    if len(idxs) != len(choices):
        raise AssertionError()
    result = []
    for idx, choice in zip(idxs, choices):
        value_idxs, value_list = vals[choice]
        result.append(value_list[list(value_idxs).index(idx)])
    return result


@scope.define_pure
def idxs_map(idxs, cmd, *args, **kwargs):
    arg_maps = []
    for arg_idxs, arg_vals in args:
        if len(arg_idxs):
            arg_maps.append(dict(zip(arg_idxs, arg_vals)))
        else:
            arg_maps.append({})

    kw_maps = {}
    for name, (arg_idxs, arg_vals) in kwargs.items():
        if len(arg_idxs):
            kw_maps[name] = dict(zip(arg_idxs, arg_vals))
        else:
            kw_maps[name] = {}

    fn = scope._impls[cmd]
    result = []

    for idx in idxs:
        try:
            positional = [mapping[idx] for mapping in arg_maps]
        except Exception:
            ERR("args_nn %s" % cmd)
            ERR("ii %s" % idx)
            ERR("arg_imap %s" % str(arg_maps))
            ERR("args_imap %s" % str(arg_maps))
            raise

        try:
            named = {name: mapping[idx] for name, mapping in kw_maps.items()}
        except Exception:
            ERR("args_nn %s" % cmd)
            ERR("ii %s" % idx)
            ERR("kw %s" % name)
            ERR("arg_imap %s" % str(arg_maps))
            raise

        try:
            result.append(fn(*positional, **named))
        except Exception:
            ERR("error calling impl of %s" % cmd)
            raise

    return result


@scope.define_pure
def idxs_take(idxs, vals, which):
    if len(idxs) != len(vals):
        raise AssertionError()
    lookup = dict(zip(idxs, vals))
    return np.asarray([lookup[idx] for idx in which])


@scope.define_pure
def uniq(lst):
    seen = set()
    result = []
    for item in lst:
        marker = id(item)
        if marker not in seen:
            seen.add(marker)
            result.append(item)
    return result


def _stochastic_vector_arg(arg, requested_idxs):
    if arg.name != "pos_args" or len(arg.pos_args) != 2:
        raise AssertionError()

    arg_idxs, arg_vals = arg.pos_args

    if arg_vals.name == "idxs_take":
        vals_node = arg_vals.arg["vals"]
        if vals_node.name == "asarray":
            inputs = vals_node.inputs()
            if inputs and inputs[0].name == "repeat":
                return inputs[0].inputs()[1]

    if arg_vals.name == "asarray":
        inputs = arg_vals.inputs()
        if inputs and inputs[0].name == "repeat":
            return inputs[0].inputs()[1]

    if arg_idxs is requested_idxs:
        return arg_vals

    raise NotImplementedError()


def vectorize_stochastic(orig):
    if orig.name != "idxs_map" or orig.pos_args[1]._obj not in stoch:
        return orig

    idxs = orig.pos_args[0]
    distribution = orig.pos_args[1]._obj

    positional = [
        _stochastic_vector_arg(arg, idxs)
        for arg in orig.pos_args[2:]
    ]
    named = [
        [name, _stochastic_vector_arg(arg, idxs)]
        for name, arg in orig.named_args
    ]

    replacement = Apply(distribution, positional, named, o_len=None)

    if "size" in dict(replacement.named_args):
        raise NotImplementedError("random node already has size")

    replacement.named_args.append(["size", scope.len(idxs)])
    return replacement


def replace_repeat_stochastic(expr, return_memo=False):
    nodes = dfs(expr)
    memo = {}

    for position, node in enumerate(nodes):
        if node.name != "idxs_map" or node.pos_args[1]._obj not in stoch:
            continue

        replacement = vectorize_stochastic(node)

        for client in nodes[position + 1 :]:
            client.replace_input(node, replacement)

        if expr is node:
            expr = replacement

        memo[node] = replacement

    if return_memo:
        return expr, memo
    return expr


class VectorizeHelper:
    def __init__(self, expr, expr_idxs, build=True):
        self.expr = expr
        self.expr_idxs = expr_idxs
        self.dfs_nodes = dfs(expr)

        self.params = {}
        for node in self.dfs_nodes:
            if node.name == "hyperopt_param":
                self.params[node.arg["label"].obj] = node.arg["obj"]

        self.idxs_memo = {}
        self.take_memo = {}

        self.v_expr = self.build_idxs_vals(expr, expr_idxs)
        self.assert_integrity_idxs_take()

    def assert_integrity_idxs_take(self):
        if dfs(self.expr) != self.dfs_nodes:
            raise AssertionError()

        if set(self.idxs_memo) != set(self.take_memo):
            raise AssertionError()

        for node, all_idxs in self.idxs_memo.items():
            takes = self.take_memo[node]
            values = takes[0].pos_args[1]
            if all_idxs.name != "array_union":
                raise AssertionError()

            for take in takes:
                if take.name != "idxs_take":
                    raise AssertionError()
                if take.pos_args[:2] != [all_idxs, values]:
                    raise AssertionError()

    def _memoized_take(self, node, wanted_idxs):
        all_idxs = self.idxs_memo[node]
        all_idxs.pos_args.append(wanted_idxs)
        values = self.take_memo[node][0].pos_args[1]
        take = scope.idxs_take(all_idxs, values, wanted_idxs)
        self.take_memo[node].append(take)
        return scope.pos_args(wanted_idxs, take)

    def _finish_node(self, node, all_idxs, values, wanted_idxs):
        take = scope.idxs_take(all_idxs, values, wanted_idxs)
        self.take_memo[node] = [take]
        return scope.pos_args(wanted_idxs, take)

    def build_idxs_vals(self, node, wanted_idxs):
        if node in self.idxs_memo:
            return self._memoized_take(node, wanted_idxs)

        all_idxs = scope.array_union(wanted_idxs)
        self.idxs_memo[node] = all_idxs

        if node.name == "literal":
            values = scope.asarray(scope.repeat(node, scope.len(all_idxs)))

        elif node.name == "switch":
            selector_pair = self.build_idxs_vals(node.pos_args[0], all_idxs)
            selector_vals = selector_pair.pos_args[1]

            n_choices = len(node.pos_args) - 1
            split_idxs = scope.vchoice_split(
                all_idxs,
                selector_vals,
                n_choices,
            )

            option_pairs = []
            for option_num, option in enumerate(node.pos_args[1:]):
                option_idxs = scope.getitem(split_idxs, option_num)
                option_pairs.append(self.build_idxs_vals(option, option_idxs))

            values = scope.vchoice_merge(
                all_idxs,
                selector_vals,
                *option_pairs
            )

        else:
            positional = [
                self.build_idxs_vals(arg, all_idxs)
                for arg in node.pos_args
            ]
            named = {
                name: self.build_idxs_vals(arg, all_idxs)
                for name, arg in node.named_args
            }
            values = scope.idxs_map(
                all_idxs,
                node.name,
                *positional,
                **named
            )

        return self._finish_node(node, all_idxs, values, wanted_idxs)


def vectorize(expr, expr_idxs):
    helper = VectorizeHelper(expr, expr_idxs)
    return replace_repeat_stochastic(helper.v_expr)