# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg224::fnmatch.fnmatch+os.walk
# name: fnmatch_os_primitive
# summary: Uses fnmatch.fnmatch, os.walk across 3 repos
# anchor_symbols: ['fnmatch.fnmatch', 'os.walk']
# observed in 3 repos: ['allenai__allennlp', 'business-science__ai-data-science-team', 'ruc-datalab__DeepAnalyze']...

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/tests/run.py::gather_test_suites_files ---
def gather_test_suites_files(test_dir, pattern):
    case_file_list = []
    for dirpath, dirnames, filenames in os.walk(test_dir):
        for file in filenames:
            if fnmatch(file, pattern):
                case_file_list.append(file)

    return case_file_list

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/tests/run.py::gather_test_cases ---
def gather_test_cases(test_dir, pattern, list_tests):
    case_list = []
    for dirpath, dirnames, filenames in os.walk(test_dir):
        for file in filenames:
            if fnmatch(file, pattern):
                case_list.append(file)

    test_suite = unittest.TestSuite()

    for case in case_list:
        test_case = unittest.defaultTestLoader.discover(
            start_dir=test_dir, pattern=case
        )
        test_suite.addTest(test_case)
        if hasattr(test_case, "__iter__"):
            for subcase in test_case:
                if list_tests:
                    print(subcase)
        else:
            if list_tests:
                print(test_case)
    return test_suite

# --- from allenai__allennlp::allennlp/common/file_utils.py::_find_entries ---
def _find_entries(
    patterns: List[str] = None,
    cache_dir: Union[str, Path] = None,
) -> Tuple[int, Dict[str, _CacheEntry]]:
    """
    Find all cache entries, filtering ones that don't match any of the glob patterns given.

    Returns the total size of the matching entries and mapping or resource name to meta data.

    The values in the returned mapping are tuples because we seperate meta entries that
    correspond to extraction directories vs regular cache entries.
    """
    cache_dir = os.path.expanduser(cache_dir or CACHE_DIRECTORY)

    total_size: int = 0
    cache_entries: Dict[str, _CacheEntry] = defaultdict(lambda: _CacheEntry([], []))
    for meta_path in glob.glob(str(cache_dir) + "/*.json"):
        meta = _Meta.from_path(meta_path)
        if patterns and not any(fnmatch(meta.resource, p) for p in patterns):
            continue
        if meta.extraction_dir:
            cache_entries[meta.resource].extraction_dirs.append(meta)
        else:
            cache_entries[meta.resource].regular_files.append(meta)
        total_size += meta.size

    # Sort entries for each resource by creation time, newest first.
    for entry in cache_entries.values():
        entry.regular_files.sort(key=lambda meta: meta.creation_time, reverse=True)
        entry.extraction_dirs.sort(key=lambda meta: meta.creation_time, reverse=True)

    return total_size, cache_entries

# --- from business-science__ai-data-science-team::ai_data_science_team/tools/data_loader.py::search_files_by_pattern ---
def search_files_by_pattern(
    directory_path: str = os.getcwd(), pattern: str = "*.csv", recursive: bool = False
) -> Tuple[str, List[Dict]]:
    """
    Tool: search_files_by_pattern
    Description:
        Searches for files (optionally in subdirectories) that match a given
        wildcard pattern (e.g. "*.csv", "*.xlsx", etc.), returning a tuple:
          (1) content (str): A multiline summary of the matched files.
          (2) artifact (List[Dict]): A list of dicts with file path info.

    Args:
        directory_path (str): Directory path to start searching from.
        pattern (str): A wildcard pattern, e.g. "*.csv". Default is "*.csv".
        recursive (bool): Whether to search in subdirectories. Default is False.

    Returns:
        Tuple[str, List[Dict]]:
            content: A user-friendly string showing matched file paths.
            artifact: A list of dictionaries, each representing a matched file.

    Example:
        content, artifact = search_files_by_pattern("/path/to/folder", "*.csv", recursive=True)
    """
    print(f"    * Tool: search_files_by_pattern | {directory_path}")

    import os
    import fnmatch

    try:
        base_path = Path(directory_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid directory path: {exc}", []

    if not base_path.is_dir():
        return f"Directory not found: {base_path}", []

    matched_files = []
    if recursive:
        for root, dirs, files in os.walk(base_path):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    matched_files.append(os.path.join(root, filename))
    else:
        # Non-recursive
        for filename in os.listdir(base_path):
            full_path = os.path.join(base_path, filename)
            if os.path.isfile(full_path) and fnmatch.fnmatch(filename, pattern):
                matched_files.append(full_path)

    # Create a human-readable summary (content)
    if matched_files:
        lines = [f"Found {len(matched_files)} file(s) matching '{pattern}':"]
        for f in matched_files:
            lines.append(f" - {f}")
        content = "\n".join(lines)
    else:
        content = f"No files found matching '{pattern}'."

    # Create artifact as a list of dicts for DataFrame conversion
    artifact = [{"file_path": path} for path in matched_files]

    return content, artifact
