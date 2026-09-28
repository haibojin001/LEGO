from collections import deque

from .types import is_seqcont

__all__ = ['tree_leaves', 'ltree_leaves', 'tree_nodes', 'ltree_nodes']


def tree_leaves(root, follow=is_seqcont, children=iter):
    """Yield the terminal values of a tree."""
    pending = deque([(root,)])

    while pending:
        current = iter(pending.pop())

        for value in current:
            if follow(value):
                pending.append(current)
                pending.append(children(value))
                break
            yield value


def ltree_leaves(root, follow=is_seqcont, children=iter):
    """Return the terminal values of a tree as a list."""
    return list(tree_leaves(root, follow, children))


def tree_nodes(root, follow=is_seqcont, children=iter):
    """Yield every value in a tree in depth-first order."""
    pending = deque([(root,)])

    while pending:
        current = iter(pending.pop())

        for value in current:
            yield value
            if follow(value):
                pending.append(current)
                pending.append(children(value))
                break


def ltree_nodes(root, follow=is_seqcont, children=iter):
    """Return every value in a tree as a list."""
    return list(tree_nodes(root, follow, children))