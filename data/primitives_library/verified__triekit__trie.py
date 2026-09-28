class _TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self._root = _TrieNode()
        self._size = 0

    def insert(self, word: str):
        node = self._root
        for char in word:
            child = node.children.get(char)
            if child is None:
                child = _TrieNode()
                node.children[char] = child
            node = child

        if not node.is_end:
            node.is_end = True
            self._size += 1

    def search(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None

    def delete(self, word: str) -> bool:
        path = []
        node = self._root

        for char in word:
            child = node.children.get(char)
            if child is None:
                return False
            path.append((node, char, child))
            node = child

        if not node.is_end:
            return False

        node.is_end = False
        self._size -= 1

        for parent, char, child in reversed(path):
            if child.is_end or child.children:
                break
            del parent.children[char]

        return True

    def keys_with_prefix(self, prefix: str) -> list:
        node = self._find_node(prefix)
        if node is None:
            return []

        results = []

        def collect(current_node, current_word):
            if current_node.is_end:
                results.append(current_word)

            for char in sorted(current_node.children):
                collect(current_node.children[char], current_word + char)

        collect(node, prefix)
        return results

    def _find_node(self, text: str):
        node = self._root
        for char in text:
            node = node.children.get(char)
            if node is None:
                return None
        return node