# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg842::re.escape+re.findall
# name: re_primitive
# summary: Uses re.escape, re.findall across 2 repos
# anchor_symbols: ['re.escape', 're.findall']
# observed in 2 repos: ['HunterMcGushion__hyperparameter_hunter', 'pgalko__BambooAI']...

# --- from HunterMcGushion__hyperparameter_hunter::hyperparameter_hunter/compat/keras_optimization_helper.py::find_space_fragments ---
def find_space_fragments(string):
    """Locate and name all hyperparameter choice declaration fragments in `string`

    Parameters
    ----------
    string: String
        A string assumed to be the source code of a Keras model-building function, in which
        hyperparameter choice declaration strings may be found

    Returns
    -------
    clipped_choices: List
        All hyperparameter choice declaration strings found in `string` - in order of appearance
    names: List
        The names of all hyperparameter choice declarations in `string` - in order of appearance
    start_indexes: List
        The indexes at which each hyperparameter choice declaration string was found in `string` -
        in order of appearance

    Examples
    --------
    >>> find_space_fragments("foo")
    ([], [], [])"""
    try:
        unclipped_choices, start_indexes = zip(*iter_fragments(string, is_match=is_space_match))
    except ValueError:
        return [], [], []
    clipped_choices = []
    names = []

    for choice in unclipped_choices:
        name = re.findall(r"(\w+(?=\s*[=(]\s*" + re.escape(choice) + r"))", string)
        # FLAG: Might need to prepend name with "_" to prevent possible duplicate extra params
        names.append(name[0] if (len(name) > 0) else names[-1])
        clipped_choices.append(clean_parenthesized_string(choice))

    #################### Fix Duplicated Names ####################
    for i in list(range(len(names)))[::-1]:
        duplicates = [_ for _ in names[0:i] if _ == names[i]]
        names[i] += "_{}".format(len(duplicates)) if len(duplicates) > 0 else ""

    return clipped_choices, names, list(start_indexes)

# --- from pgalko__BambooAI::bambooai/messages/reg_ex.py::_extract_code ---
def _extract_code(response: str, analyst: str, provider: str) -> str:
    """Extract and sanitize code while preserving comments and structure."""
    blacklist = [
        'subprocess', 'sys', 'exec', 'socket', 'urllib',
        'shutil', 'pickle', 'ctypes', 'multiprocessing', 'tempfile', 'glob', 'pty',
        'commands', 'cgi', 'cgitb', 'xml.etree.ElementTree', 'builtins'
    ]
    
    # Extract code from markdown blocks
    # Replace <|im_sep|> with ``` to match the markdown code block syntax
    response = re.sub(re.escape("<|im_sep|>"), "```", response)
    # Find all code segments enclosed in triple backticks with "python"
    code_segments = re.findall(r'```python\n(\s*.*?)\s*```', response, re.DOTALL)
    # If no segments found, try without "python"
    if not code_segments:
      code_segments = re.findall(r'```(?:python\n|\n)(\s*.*?)\s*```', response, re.DOTALL)

    if not code_segments:
      return ""
    
    # Normalize the indentation for each code segment
    normalized_code_segments = [_normalize_indentation(segment) for segment in code_segments]

    # Combine the normalized code segments into a single string
    code = '\n\n'.join(normalized_code_segments).lstrip()

    # Split the code into lines
    lines = code.splitlines()
    
    # Process the code line by line
    processed_lines = []
    main_start, main_end, main_indent = find_main_block(code)
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Handle empty lines
        if not line.strip():
            processed_lines.append(line)
            i += 1
            continue
        
        # Handle main block
        if main_start and i == main_start - 1:  # We're at the if __name__ line
            main_content = process_main_block(lines, main_start, main_end, main_indent, blacklist)
            processed_lines.extend(main_content)
            i = main_end
            continue
            
        # Handle blacklisted imports
        pattern = r"\b(" + "|".join(blacklist) + r")\b" # Match whole words
        if re.search(pattern, line):
            processed_lines.append(f"# not allowed {line}")
            i += 1
            continue
            
        # Handle transformations
        if 'plt.savefig' in line:
            indent = len(line) - len(line.lstrip())
            line = ' ' * indent + 'plt.show()'
        #line = re.sub(r"df\s*=\s*pd\.read_csv\((.*?)\)", "", line) # Temporary disable #TODO
        line = re.sub(r'plt\.style\.use\s*\(\s*\'seaborn\'\s*\)', 'sns.set_style("whitegrid")', line)
        
        if analyst == "Data Analyst DF" and provider == "local":
            if re.search(r"data=pd\.", line):
                line = re.sub(r"\bdata\b", "df", line)
            line = re.sub(
                r"(?<![a-zA-Z0-9_-])df\s*=\s*pd\.DataFrame\((.*?)\)",
                "# The dataframe df has already been defined",
                line
            )
        
        processed_lines.append(line)
        i += 1
    
    # Clean up multiple empty lines
    result = '\n'.join(processed_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()
