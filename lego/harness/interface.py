"""Public-interface stubs for target modules.

A LEGO-REPO task reconstructs a package from its public interface, so the
construction pipeline must never see an original implementation. ``stub`` turns a
module's source into what a caller of that module can observe: imports, the
module docstring, ``__all__``, class and function signatures with decorators and
docstrings, and the names of module-level constants. Every function body is
replaced by ``...``.

``interface_text`` is the single entry point used by the pipeline; the mode is a
run setting (``RunConfig.interface``):

  stub    signatures + docstrings only (default)
  source  the raw original module, truncated. Only for reproducing pre-r3 records;
          it exposes the implementation and is not a valid LEGO-REPO setting.
"""

from __future__ import annotations

import ast

MAX_CHARS = 9000


class _Stubber(ast.NodeTransformer):
    def __init__(self, import_safe: bool = False):
        self.import_safe = import_safe

    def _strip_body(self, node):
        if self.import_safe:
            node.decorator_list = []
        doc = ast.get_docstring(node, clean=False)
        body = []
        if doc is not None:
            body.append(ast.Expr(ast.Constant(doc)))
        body.append(ast.Expr(ast.Constant(Ellipsis)))
        node.body = body
        return node

    def visit_FunctionDef(self, node):
        return self._strip_body(node)

    def visit_AsyncFunctionDef(self, node):
        return self._strip_body(node)

    def visit_ClassDef(self, node):
        if self.import_safe:
            node.decorator_list = []
        doc = ast.get_docstring(node, clean=False)
        body = [ast.Expr(ast.Constant(doc))] if doc is not None else []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body.append(self._strip_body(item))
            elif isinstance(item, ast.ClassDef):
                body.append(self.visit_ClassDef(item))
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                body.append(_constant_stub(item))
        node.body = body or [ast.Expr(ast.Constant(Ellipsis))]
        return node


def _is_simple_literal(node) -> bool:
    try:
        ast.literal_eval(node)
    except Exception:
        return False
    return len(ast.unparse(node)) <= 80


def _constant_stub(node):
    """Keep a module/class attribute's name (and annotation); keep the value only
    for short literals such as ``__version__`` or enum-like flags."""
    if isinstance(node, ast.AnnAssign):
        if node.value is not None and not _is_simple_literal(node.value):
            node.value = ast.Constant(Ellipsis)
        return node
    if not _is_simple_literal(node.value):
        node.value = ast.Constant(Ellipsis)
    return node


def stub(source: str, import_safe: bool = False) -> str:
    """Interface stub of one module. Falls back to the import lines and the
    top-level ``def``/``class`` headers if the module does not parse.

    ``import_safe`` drops decorators so that the stub can stand in for a
    not-yet-generated sibling during carried validation."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        keep = [l for l in source.splitlines()
                if l.startswith(("import ", "from ", "def ", "class ",
                                 "async def "))]
        return "\n".join(keep)
    body = []
    doc = ast.get_docstring(tree, clean=False)
    if doc is not None:
        body.append(ast.Expr(ast.Constant(doc)))
    st = _Stubber(import_safe)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            body.append(st.visit(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            body.append(_constant_stub(node))
        elif isinstance(node, ast.If) and _is_type_checking(node):
            body.append(node)          # TYPE_CHECKING imports are interface
        elif isinstance(node, ast.Try):
            imps = [n for n in node.body
                    if isinstance(n, (ast.Import, ast.ImportFrom))]
            body.extend(imps)          # optional-import blocks: keep the imports
    tree.body = body
    return ast.unparse(ast.fix_missing_locations(tree))


def _is_type_checking(node: ast.If) -> bool:
    t = node.test
    return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
        isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING")


def exports(source: str) -> list[str]:
    """Public top-level names of a module (``__all__`` if declared)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            try:
                return [str(x) for x in ast.literal_eval(n.value)]
            except Exception:
                break
    names = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(n.name)
        elif isinstance(n, ast.Assign):
            names += [t.id for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.append(n.target.id)
    return [x for x in names if not x.startswith("_") or x.startswith("__")]


def internal_imports(source: str, pkg: str, module_path: str) -> list[str]:
    """Package-internal modules this module imports, as paths relative to the
    package root (``sub/mod.py``). Used for the dependency order of Pi_t."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    here = module_path.replace("\\", "/").split("/")[:-1]
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.level:
                base = here[: len(here) - (n.level - 1)] if n.level > 1 else here
                parts = base + (n.module.split(".") if n.module else [])
            elif n.module and (n.module == pkg or n.module.startswith(pkg + ".")):
                parts = n.module.split(".")[1:]
            else:
                continue
            out.add("/".join(parts))
            for a in n.names:
                out.add("/".join(parts + [a.name]))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name == pkg or a.name.startswith(pkg + "."):
                    out.add("/".join(a.name.split(".")[1:]))
    return sorted(out)


def external_imports(source: str, pkg: str) -> list[str]:
    """Top-level distributions a module imports from outside its package."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
            out.add(n.module.split(".")[0])
    out.discard(pkg)
    return sorted(out)


def interface_text(source: str, mode: str = "stub",
                   max_chars: int = MAX_CHARS) -> str:
    if mode == "source":
        return source[:max_chars]
    if mode != "stub":
        raise ValueError(f"unknown interface mode {mode!r}")
    return stub(source)[:max_chars]
