__all__ = ["AhoCorasick"]


class _Node:
    __slots__ = ("children", "fail", "output")

    def __init__(self):
        self.children = {}
        self.fail = 0
        self.output = []


class AhoCorasick:
    def __init__(self, patterns: list):
        if isinstance(patterns, (str, bytes)):
            raise TypeError("patterns must be an iterable of strings, not a string")

        try:
            pattern_list = list(patterns)
        except TypeError as exc:
            raise TypeError("patterns must be an iterable of strings") from exc

        for pattern in pattern_list:
            if not isinstance(pattern, str):
                raise TypeError("all patterns must be strings")

        self._patterns = pattern_list
        self._nodes = [_Node()]

        for pattern_index, pattern in enumerate(pattern_list):
            state = 0
            for char in pattern:
                children = self._nodes[state].children
                next_state = children.get(char)
                if next_state is None:
                    next_state = len(self._nodes)
                    children[char] = next_state
                    self._nodes.append(_Node())
                state = next_state
            self._nodes[state].output.append((pattern_index, pattern))

        self._build_failure_links()

    def _build_failure_links(self):
        queue = []

        for child_state in self._nodes[0].children.values():
            self._nodes[child_state].fail = 0
            queue.append(child_state)

        head = 0
        while head < len(queue):
            state = queue[head]
            head += 1

            fail_state = self._nodes[state].fail
            fail_output = self._nodes[fail_state].output
            if fail_output:
                self._nodes[state].output.extend(fail_output)

            if len(self._nodes[state].output) > 1:
                self._nodes[state].output.sort(key=lambda item: item[0])

            for char, child_state in self._nodes[state].children.items():
                fallback = self._nodes[state].fail
                while fallback and char not in self._nodes[fallback].children:
                    fallback = self._nodes[fallback].fail

                self._nodes[child_state].fail = self._nodes[fallback].children.get(char, 0)
                queue.append(child_state)

    def find_all(self, text: str) -> list:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        matches = []
        state = 0
        nodes = self._nodes

        if nodes[0].output:
            for _pattern_index, pattern in nodes[0].output:
                matches.append((-1, pattern))

        for end_index, char in enumerate(text):
            while state and char not in nodes[state].children:
                state = nodes[state].fail

            state = nodes[state].children.get(char, 0)

            if nodes[state].output:
                for _pattern_index, pattern in nodes[state].output:
                    matches.append((end_index, pattern))

        return matches