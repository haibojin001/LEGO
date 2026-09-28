import os
from pathlib import Path

from grep_ast import TreeContext
from grep_ast.parsers import PARSERS, filename_to_lang
from tree_sitter_languages import get_language, get_parser

from cover_agent.lsp_logic.file_map.queries.get_queries import get_queries_scheme


class FileMap:
    """
    This class is used to summarize the content of a file using tree-sitter queries.
    Supported languages: C, C++, C#, elisp, elixir, go, java, javascript, ocaml,
    php, python, ql, ruby, rust, typescript.
    """

    def __init__(
        self,
        fname_full_path: str,
        parent_context=True,
        child_context=False,
        header_max=0,
        margin=0,
        project_base_path: str = None,
    ):
        self.fname_full_path = fname_full_path
        self.project_base_path = project_base_path

        if project_base_path:
            self.fname_rel = os.path.relpath(fname_full_path, project_base_path)
        else:
            self.fname_rel = fname_full_path

        self.main_queries_path = Path(__file__).parent.parent / "queries"

        if not os.path.exists(fname_full_path):
            print(f"File {fname_full_path} does not exist")

        with open(fname_full_path, "r") as source_file:
            file_contents = source_file.read()

        self.code = file_contents.rstrip("\n") + "\n"
        self.parent_context = parent_context
        self.child_context = child_context
        self.header_max = header_max
        self.margin = margin

    def summarize(self):
        results = self.get_query_results()
        return self.query_processing(results)

    def render_file_summary(self, lines_of_interest: list):
        tree_context = TreeContext(
            self.fname_rel,
            self.code,
            color=False,
            line_number=True,
            parent_context=self.parent_context,
            child_context=self.child_context,
            last_line=False,
            margin=self.margin,
            mark_lois=False,
            loi_pad=0,
            header_max=self.header_max,
            show_top_of_file_parent_scope=False,
        )

        tree_context.lines_of_interest = set()
        tree_context.add_lines_of_interest(lines_of_interest)
        tree_context.add_context()
        return tree_context.format()

    def query_processing(self, query_results: list):
        if not query_results:
            return ""

        definition_lines = [
            item["line"] for item in query_results if item["kind"] == "def"
        ]

        return (
            "\n"
            + query_results[0]["fname"]
            + ":\n"
            + self.render_file_summary(definition_lines)
        )

    def get_query_results(self):
        language_name = filename_to_lang(self.fname_rel)
        if not language_name:
            return

        try:
            language = get_language(language_name)
            parser = get_parser(language_name)
        except Exception as error:
            print(f"Skipping file {self.fname_rel}: {error}")
            return

        query_text = get_queries_scheme(language_name)
        syntax_tree = parser.parse(self.code.encode("utf-8"))
        query = language.query(query_text)
        captures = list(query.captures(syntax_tree.root_node))

        seen_kinds = set()
        results = []

        for node, capture_name in captures:
            if capture_name.startswith("name.definition."):
                result_kind = "def"
            elif capture_name.startswith("name.reference."):
                result_kind = "ref"
            else:
                continue

            seen_kinds.add(result_kind)
            results.append(
                {
                    "fname": self.fname_rel,
                    "name": node.text.decode("utf-8"),
                    "kind": result_kind,
                    "line": node.start_point[0],
                }
            )

        if "ref" in seen_kinds:
            return results, captures

        if "def" not in seen_kinds:
            return results, captures

        return results, captures