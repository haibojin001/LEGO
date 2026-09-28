# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg201::ast.iter_child_nodes+ast.parse
# name: ast_primitive
# summary: Uses ast.iter_child_nodes, ast.parse across 2 repos
# anchor_symbols: ['ast.iter_child_nodes', 'ast.parse']
# observed in 2 repos: ['HunterMcGushion__hyperparameter_hunter', 'microsoft__RD-Agent']...

# --- from microsoft__RD-Agent::rdagent/utils/repo/repo_utils.py::RepoAnalyzer._summarize_function ---
def _summarize_function(
        self, node: ast.FunctionDef, verbose_level: int, doc_str_level: int, sign_level: int, indent: str = ""
    ) -> str:
        summary = f"{indent}Function: {node.name}\n"
        if sign_level > 0:
            # Generate the function signature
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")

            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            signature = f"{node.name}({', '.join(args)}){returns}"
            summary += f"{indent}  Signature: {signature}\n"

        if doc_str_level > 0 and ast.get_docstring(node):
            doc = ast.get_docstring(node)
            summary += f"{indent}  Purpose: {doc.split('.')[0]}.\n"
        return summary

# --- from microsoft__RD-Agent::rdagent/utils/repo/repo_utils.py::RepoAnalyzer._summarize_file ---
def _summarize_file(self, file_path: Path, verbose_level: int, doc_str_level: int, sign_level: int) -> str:
        with open(file_path, "r") as f:
            content = f.read()

        tree = ast.parse(content)
        summary = f"File: {file_path.relative_to(self.repo_path)}\n"
        summary += f"{'-' * 40}\n"

        classes = [node for node in ast.iter_child_nodes(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.iter_child_nodes(tree) if isinstance(node, ast.FunctionDef)]

        if classes:
            summary += f"This file contains {len(classes)} class{'es' if len(classes) > 1 else ''}.\n"
        if functions:
            summary += f"This file contains {len(functions)} top-level function{'s' if len(functions) > 1 else ''}.\n"

        for node in classes + functions:
            if isinstance(node, ast.ClassDef):
                summary += self._summarize_class(node, verbose_level, doc_str_level, sign_level)
            elif isinstance(node, ast.FunctionDef):
                summary += self._summarize_function(node, verbose_level, doc_str_level, sign_level)

        return summary

# --- from HunterMcGushion__hyperparameter_hunter::hyperparameter_hunter/feature_engineering.py::get_engineering_step_params ---
def get_engineering_step_params(f: callable) -> Tuple[str]:
    """Verify that callable `f` requests valid input parameters, and returns a tuple of the same
    parameters, with the assumption that the parameters are modified by `f`

    Parameters
    ----------
    f: Callable
        Feature engineering step function that requests, modifies, and returns datasets

    Returns
    -------
    Tuple
        Argument/return value names declared by `f`

    Examples
    --------
    >>> def impute_negative_one(all_inputs):
    ...     all_inputs.fillna(-1, inplace=True)
    ...     return all_inputs
    >>> get_engineering_step_params(impute_negative_one)
    ('all_inputs',)
    >>> def standard_scale(train_inputs, non_train_inputs):
    ...     scaler = StandardScaler()
    ...     train_inputs[train_inputs.columns] = scaler.fit_transform(train_inputs.values)
    ...     non_train_inputs[train_inputs.columns] = scaler.transform(non_train_inputs.values)
    ...     return train_inputs, non_train_inputs
    >>> get_engineering_step_params(standard_scale)
    ('train_inputs', 'non_train_inputs')
    >>> def error_invalid_dataset(train_inputs, foo):
    ...     return train_inputs, foo
    >>> get_engineering_step_params(error_invalid_dataset)
    Traceback (most recent call last):
        File "feature_engineering.py", line ?, in get_engineering_step_params
    ValueError: Invalid dataset name: 'foo'"""
    valid_datasets = MERGED_DATASET_NAMES + STANDARD_DATASET_NAMES
    source_code = getsource(f)
    tree = ast.parse(source_code)

    #################### Add Links to Nodes' Parents ####################
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    #################### Collect Parameters and Returns ####################
    parser = ParameterParser()
    parser.visit(tree)

    for name in parser.args:
        if name not in valid_datasets:
            raise ValueError(f"Invalid dataset name: {name!r}")
        if name.endswith("_data"):
            raise ValueError(
                f"Sorry, 'data'-suffixed parameters like {name!r} are not supported yet. "
                "Try using both the 'inputs' and 'targets' params for this dataset, instead!"
            )

    return tuple(parser.args)
