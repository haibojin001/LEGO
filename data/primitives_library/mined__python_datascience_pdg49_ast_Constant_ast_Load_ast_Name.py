# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg49::ast.Constant+ast.Load+ast.Name
# name: ast_primitive
# summary: Uses ast.Constant, ast.Load, ast.Name, ast.parse across 2 repos
# anchor_symbols: ['ast.Constant', 'ast.Load', 'ast.Name', 'ast.parse']
# observed in 2 repos: ['libffcv__ffcv', 'sinaptik-ai__pandas-ai']...

# --- from libffcv__ffcv::ffcv/pipeline/graph.py::DecoderNode.func_call_ast ---
def func_call_ast(self):
        tree = super().func_call_ast
        tree.value.args.extend([
            ast.Subscript(value=ast.Name(id='metadata', ctx=ast.Load()),
                          slice=ast.Index(value=ast.Constant(value=f'f{self.f_ix}', kind=None)), ctx=ast.Load()),
                 ast.Name(id='storage_state', ctx=ast.Load()),
        ])

        return tree

# --- from libffcv__ffcv::ffcv/pipeline/graph.py::Node.func_call_ast ---
def func_call_ast(self):
        pipeline_identifier = f'code_{self.id}'
        memory_identifier = f'memory_{self.id}'

        tree = ast.parse(f"""
{self.result_id} = {pipeline_identifier}({self.arg_id}, {memory_identifier})
        """).body[0]

        if self.with_indices:
            tree.value.args.extend([
                ast.Name(id='batch_indices', ctx=ast.Load()),
            ])
        return tree

# --- from sinaptik-ai__pandas-ai::tests/unit_tests/core/code_generation/test_code_cleaning.py::TestCodeCleaner.test_validate_and_make_table_name_case_sensitive ---
def test_validate_and_make_table_name_case_sensitive(self):
        node = ast.Assign(
            targets=[ast.Name(id="query", ctx=ast.Store())],
            value=ast.Constant(value="SELECT * FROM my_table"),
        )
        mock_dataframe = MagicMock(spec=object)
        mock_dataframe.name = "my_table"
        self.cleaner.context.dfs = [mock_dataframe]
        mock_dataframe.schema = MagicMock()
        mock_dataframe.schema.name = "my_table"
        mock_dataframe.get_dialect = MagicMock(return_value="duckdb")
        updated_node = self.cleaner._validate_and_make_table_name_case_sensitive(node)
        self.assertEqual(updated_node.value.value, "SELECT * FROM my_table")
