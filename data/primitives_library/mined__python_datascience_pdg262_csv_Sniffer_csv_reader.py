# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg262::csv.Sniffer+csv.reader
# name: csv_primitive
# summary: Uses csv.Sniffer, csv.reader across 2 repos
# anchor_symbols: ['csv.Sniffer', 'csv.reader']
# observed in 2 repos: ['modin-project__modin', 'wooey__Wooey']...

# --- from wooey__Wooey::wooey/backend/utils.py::test_delimited ---
def test_delimited(filepath):
    import csv

    with open(filepath, "r", newline="") as csv_file:
        try:
            dialect = csv.Sniffer().sniff(csv_file.read(1024 * 16), delimiters=",\t")
        except Exception as e:
            return False, None
        csv_file.seek(0)
        reader = csv.reader(csv_file, dialect)
        rows = []
        try:
            for index, entry in enumerate(reader):
                rows.append(entry)

        except Exception as e:
            return False, None

        # If > 10 rows, generate preview by slicing top and bottom 5
        # ? this might not be a great idea for massive files
        if len(rows) > 10:
            rows = rows[:5] + [None] + rows[-5:]
        # FIXME: This should be more intelligent:
        # for small files (<1000 rows?) we should take top and bottom preview 10
        # for large files we should give up and present top 10 (11)
        # same rules should apply to columns: this will require us to discard them as they're read

    return True, rows

# --- from modin-project__modin::modin/tests/pandas/utils.py::insert_lines_to_csv ---
def insert_lines_to_csv(
    csv_name: str,
    lines_positions: list,
    lines_type: str = "blank",
    encoding: str = None,
    **csv_reader_writer_params,
):
    """Insert lines to ".csv" file.

    Parameters
    ----------
    csv_name: str
        ".csv" file that should be modified.
    lines_positions: list of ints
        Lines postions that sghould be modified (serial number
        of line - begins from 0, ends in <rows_number> - 1).
    lines_type: str
        Lines types that should be inserted to ".csv" file. Possible types:
        "blank" - empty line without any delimiters/separators,
        "bad" - lines with len(lines_data) > cols_number
    encoding: str
        Encoding type that should be used during file reading and writing.
    """
    if lines_type == "blank":
        lines_data = []
    elif lines_type == "bad":
        cols_number = len(pandas.read_csv(csv_name, nrows=1).columns)
        lines_data = [x for x in range(cols_number + 1)]
    else:
        raise ValueError(
            f"acceptable values for  parameter are ['blank', 'bad'], actually passed {lines_type}"
        )
    lines = []
    with open(csv_name, "r", encoding=encoding, newline="") as read_file:
        try:
            dialect = csv.Sniffer().sniff(read_file.read())
            read_file.seek(0)
        except Exception:
            dialect = None

        reader = csv.reader(
            read_file,
            dialect=dialect if dialect is not None else "excel",
            **csv_reader_writer_params,
        )
        counter = 0
        for row in reader:
            if counter in lines_positions:
                lines.append(lines_data)
            else:
                lines.append(row)
            counter += 1
    with open(csv_name, "w", encoding=encoding, newline="") as write_file:
        writer = csv.writer(
            write_file,
            dialect=dialect if dialect is not None else "excel",
            **csv_reader_writer_params,
        )
        writer.writerows(lines)
