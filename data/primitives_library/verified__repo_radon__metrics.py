import ast
import collections
import math

from radon.raw import analyze
from radon.visitors import ComplexityVisitor, HalsteadVisitor


HalsteadReport = collections.namedtuple(
    "HalsteadReport",
    (
        "h1 h2 N1 N2 vocabulary length calculated_length volume "
        "difficulty effort time bugs"
    ),
)
Halstead = collections.namedtuple("Halstead", "total functions")


def h_visit(code):
    return h_visit_ast(ast.parse(code))


def h_visit_ast(ast_node):
    visitor = HalsteadVisitor.from_ast(ast_node)
    overall = halstead_visitor_report(visitor)
    per_function = [
        (function_visitor.context, halstead_visitor_report(function_visitor))
        for function_visitor in visitor.function_visitors
    ]
    return Halstead(overall, per_function)


def halstead_visitor_report(visitor):
    unique_operators = visitor.distinct_operators
    unique_operands = visitor.distinct_operands
    operator_count = visitor.operators
    operand_count = visitor.operands

    vocabulary = unique_operators + unique_operands
    length = operator_count + operand_count

    if unique_operators and unique_operands:
        calculated_length = (
            unique_operators * math.log(unique_operators, 2)
            + unique_operands * math.log(unique_operands, 2)
        )
    else:
        calculated_length = 0

    volume = length * math.log(vocabulary, 2) if vocabulary else 0
    difficulty = (
        (unique_operators * operand_count) / float(2 * unique_operands)
        if unique_operands
        else 0
    )
    effort = difficulty * volume

    return HalsteadReport(
        unique_operators,
        unique_operands,
        operator_count,
        operand_count,
        vocabulary,
        length,
        calculated_length,
        volume,
        difficulty,
        effort,
        effort / 18.0,
        volume / 3000.0,
    )


def mi_compute(halstead_volume, complexity, sloc, comments):
    if any(value <= 0 for value in (halstead_volume, sloc)):
        return 100.0

    unnormalized = (
        171
        - 5.2 * math.log(halstead_volume)
        - 0.23 * complexity
        - 16.2 * math.log(sloc)
        + 50 * math.sin(math.sqrt(2.46 * math.radians(comments)))
    )
    return min(max(0.0, unnormalized * 100 / 171.0), 100.0)


def mi_parameters(code, count_multi=True):
    tree = ast.parse(code)
    raw_metrics = analyze(code)
    comment_lines = raw_metrics.comments
    if count_multi:
        comment_lines += raw_metrics.multi

    comment_percent = (
        comment_lines / float(raw_metrics.sloc) * 100 if raw_metrics.sloc else 0
    )

    return (
        h_visit_ast(tree).total.volume,
        ComplexityVisitor.from_ast(tree).total_complexity,
        raw_metrics.lloc,
        comment_percent,
    )


def mi_visit(code, multi):
    return mi_compute(*mi_parameters(code, multi))


def mi_rank(score):
    return chr(65 + (9 - score >= 0) + (19 - score >= 0))