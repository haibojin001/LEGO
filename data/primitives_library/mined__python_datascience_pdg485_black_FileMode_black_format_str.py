# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg485::black.FileMode+black.format_str
# name: black_primitive
# summary: Uses black.FileMode, black.format_str across 2 repos
# anchor_symbols: ['black.FileMode', 'black.format_str']
# observed in 2 repos: ['WecoAI__aideml', 'stellargraph__stellargraph']...

# --- from WecoAI__aideml::aide/utils/response.py::format_code ---
def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code

# --- from stellargraph__stellargraph::scripts/format_notebooks.py::FormatCodeCellPreprocessor.preprocess_cell ---
def preprocess_cell(self, cell, resources, cell_index):
        mode = FileMode(line_length=self.linelength)

        if cell.cell_type == "code":
            try:
                formatted = format_str(src_contents=cell["source"], mode=mode)
            except InvalidInput as err:
                print(f"Formatter error: {err}")
                formatted = cell["source"]

            if formatted and formatted[-1] == "\n":
                formatted = formatted[:-1]

            if cell["source"] != formatted:
                self.notebook_cells_changed += 1

            cell["source"] = formatted
        return cell, resources
