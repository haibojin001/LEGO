import math

from radon.visitors import GET_COMPLEXITY, ComplexityVisitor, code2ast


SCORE = lambda block: -GET_COMPLEXITY(block)
LINES = lambda block: block.lineno
ALPHA = lambda block: block.name


def cc_rank(cc):
    if cc < 0:
        raise ValueError('Complexity must be a non-negative value')
    rank = int(math.ceil(cc / 10.0) or 1)
    rank -= (1, 0)[5 - cc < 0]
    return chr(min(rank, 5) + 65)


def average_complexity(blocks):
    if not blocks:
        return 0
    return sum((GET_COMPLEXITY(block) for block in blocks), 0.0) / len(blocks)


def sorted_results(blocks, order=SCORE):
    return sorted(blocks, key=order)


def add_inner_blocks(blocks):
    result = []
    pending = blocks[:]

    while pending:
        block = pending.pop()
        result.append(block)

        for attribute in ('closures', 'inner_classes'):
            for child in getattr(block, attribute, ()):
                child = child._replace(name=block.name + '.' + child.name)
                pending.append(child)

                for method in getattr(child, 'methods', ()):
                    pending.append(
                        method._replace(classname=child.name)
                    )

    return result


def cc_visit(code, **kwargs):
    return cc_visit_ast(code2ast(code), **kwargs)


def cc_visit_ast(ast_node, **kwargs):
    return ComplexityVisitor.from_ast(ast_node, **kwargs).blocks