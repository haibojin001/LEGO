import io

from .pyll_utils import expr_to_config


def dot_hyperparameters(expr):
    conditions = ()
    hyperparameters = {}
    expr_to_config(expr, conditions, hyperparameters)

    output = io.StringIO()
    print("digraph {", file=output)
    emitted_edges = set()

    def emit_variable(name):
        print('"%s" [ shape=box];' % name, file=output)

    def emit_condition(name):
        print('"%s" [ shape=ellipse];' % name, file=output)

    def emit_edge(source, target):
        line = f'"{source}" -> "{target}";'
        if line not in emitted_edges:
            print(line, file=output)
            emitted_edges.add(line)

    for name, specification in list(hyperparameters.items()):
        emit_variable(name)

        for conjunction in specification["conditions"]:
            if len(conjunction) > 1:
                combined = " & ".join(
                    "%(name)s%(op)s%(val)s" % condition.__dict__
                    for condition in conjunction
                )
                emit_condition(combined)
                emit_edge(combined, name)

                for condition in conjunction:
                    label = f"{condition.name}{condition.op}{condition.val}"
                    emit_condition(label)
                    emit_edge(condition.name, label)
                    emit_edge(label, combined)
            elif len(conjunction) == 1:
                condition = conjunction[0]
                label = f"{condition.name}{condition.op}{condition.val}"
                emit_edge(condition.name, label)
                emit_condition(label)
                emit_edge(label, name)

    print("}", file=output)
    return output.getvalue()