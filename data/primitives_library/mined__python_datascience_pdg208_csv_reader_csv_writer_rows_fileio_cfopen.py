# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg208::csv.reader+csv.writer+rows.fileio.cfopen
# name: csv_rows_primitive
# summary: Uses csv.reader, csv.writer, rows.fileio.cfopen across 3 repos
# anchor_symbols: ['csv.reader', 'csv.writer', 'rows.fileio.cfopen']
# observed in 3 repos: ['allenai__allennlp', 'microsoft__RD-Agent', 'turicas__rows']...

# --- from turicas__rows::rows/cli.py::csv_split ---
def csv_split(
    input_encoding,
    output_encoding,
    buffer_size,
    quiet,
    destination_pattern,
    source,
    lines,
):
    """Split CSV into equal parts (by number of lines).

    Input and output files can be compressed.
    """
    import csv

    from rows.fileio import cfopen
    from rows.utils import COMPRESSED_EXTENSIONS

    input_encoding = input_encoding or DEFAULT_INPUT_ENCODING
    if destination_pattern is None:
        first_part, extension = source.rsplit(".", 1)
        if extension.lower() in COMPRESSED_EXTENSIONS:
            first_part, new_extension = first_part.rsplit(".", 1)
            extension = new_extension + "." + extension
        destination_pattern = first_part + "-{part:03d}." + extension

    part = 0
    output_fobj = None
    writer = None
    input_fobj = cfopen(source, encoding=input_encoding, buffering=buffer_size)
    reader = csv.reader(input_fobj)
    header = next(reader)
    if not quiet:
        reader = _tqdm_if_available(reader)
    for index, row in enumerate(reader):
        if index % lines == 0:
            if output_fobj is not None:
                output_fobj.close()
            part += 1
            output_fobj = cfopen(
                destination_pattern.format(part=part),
                mode="w",
                encoding=output_encoding,
                buffering=buffer_size,
            )
            writer = csv.writer(output_fobj)
            writer.writerow(header)
        writer.writerow(row)
    input_fobj.close()

# --- from microsoft__RD-Agent::rdagent/scenarios/finetune/datasets/financeiq/split.py::split_financeiq_dataset ---
def split_financeiq_dataset(data_dir: str, split: Literal["train", "test"]) -> None:
    """
    Iterate over CSV files in the directory and apply the split in-place.
    """
    path = Path(data_dir)

    # Process CSV files
    for f in list(path.rglob("*.csv")):
        # HACK:
        # FinanceIQ specific: 'dev' folder is small and used for few-shot.
        # We preserve it for benchmarking (split='test') but remove for training (split='train') to avoid leakage.
        # Some times, the training in debug mode of llama factory will only check few samples. Which may results in failures
        rel_parts = f.relative_to(path).parts
        if "dev" in rel_parts:
            if split == "train":
                f.unlink()
            continue

        rows = []
        header = None
        # Use 'utf-8-sig' to handle potential BOM in Excel-saved CSVs, or just 'utf-8'
        # Assuming 'utf-8' for now as it's standard for HF datasets
        with open(f, "r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            try:
                header = next(reader)
                rows = list(reader)
            except StopIteration:
                # Empty file
                continue

        indices = get_split_indices(len(rows), split)
        new_rows = rows[indices]

        with open(f, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            if header:
                writer.writerow(header)
            writer.writerows(new_rows)

# --- from allenai__allennlp::tests/commands/predict_test.py::TestPredict.test_alternative_file_formats ---
def test_alternative_file_formats(self):
        @Predictor.register("classification-csv")
        class CsvPredictor(TextClassifierPredictor):
            """same as classification predictor but using CSV inputs and outputs"""

            def load_line(self, line: str) -> JsonDict:
                reader = csv.reader([line])
                sentence, label = next(reader)
                return {"sentence": sentence, "label": label}

            def dump_line(self, outputs: JsonDict) -> str:
                output = io.StringIO()
                writer = csv.writer(output)
                row = [outputs["label"], *outputs["probs"]]

                writer.writerow(row)
                return output.getvalue()

        with open(self.infile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["the seahawks won the super bowl in 2016", "pos"])
            writer.writerow(["the mariners won the super bowl in 2037", "neg"])

        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.infile),  # input_file
            "--output-file",
            str(self.outfile),
            "--predictor",
            "classification-csv",
            "--silent",
        ]

        main()
        assert os.path.exists(self.outfile)

        with open(self.outfile) as f:
            reader = csv.reader(f)
            results = [row for row in reader]

        assert len(results) == 2
        for row in results:
            assert len(row) == 3  # label and 2 class probabilities
            label, *probs = row
            for prob in probs:
                assert 0 <= float(prob) <= 1
            assert label != ""

        shutil.rmtree(self.tempdir)

# --- from turicas__rows::rows/cli.py::csv_merge ---
def csv_merge(
    input_encoding,
    output_encoding,
    no_strip,
    no_remove_empty_lines,
    sample_size,
    buffer_size,
    sources,
    destination,
):
    import csv
    from collections import defaultdict

    from rows.fields import make_header, slug
    from rows.fileio import cfopen
    from rows.plugins import csv as rows_csv

    # TODO: add option to preserve original key names
    # TODO: add --quiet

    strip = not no_strip
    remove_empty_lines = not no_remove_empty_lines

    metadata = defaultdict(dict)
    final_header = []
    for filename in _tqdm_if_available(sources, desc="Detecting dialects and headers"):
        inspector = rows_csv.CsvInspector(filename, chunk_size=sample_size, encoding=input_encoding)
        metadata[filename]["dialect"] = inspector.dialect

        # Get header
        # TODO: fix final header in case of empty field names (a command like
        # `rows csv-clean` would fix the problem if run before `csv-merge` for
        # each file).
        metadata[filename]["fobj"] = cfopen(filename, encoding=inspector.encoding, buffering=buffer_size)
        metadata[filename]["reader"] = csv.reader(metadata[filename]["fobj"], dialect=metadata[filename]["dialect"])
        metadata[filename]["header"] = make_header(next(metadata[filename]["reader"]))
        metadata[filename]["header_map"] = {}
        for field_name in metadata[filename]["header"]:
            field_name_slug = slug(field_name)
            metadata[filename]["header_map"][field_name_slug] = field_name
            if field_name_slug not in final_header:
                final_header.append(field_name_slug)
    # TODO: is it needed to use make_header here?

    progress_bar = _tqdm_if_available(desc="Exporting data") if _tqdm_available else None
    output_fobj = cfopen(destination, mode="w", encoding=output_encoding, buffering=buffer_size)
    writer = csv.writer(output_fobj)
    writer.writerow(final_header)
    for index, filename in enumerate(sources):
        if progress_bar is not None:
            progress_bar.desc = "Exporting data {}/{}".format(index + 1, len(sources))
        meta = metadata[filename]
        field_indexes = [
            meta["header"].index(field_name) if field_name in meta["header"] else None for field_name in final_header
        ]
        if strip:

            def create_new_row(row):
                return [row[index].strip() if index is not None else None for index in field_indexes]

        else:

            def create_new_row(row):
                return [row[index] if index is not None else None for index in field_indexes]

        for row in meta["reader"]:
            new_row = create_new_row(row)
            if remove_empty_lines and not any(new_row):
                continue
            writer.writerow(new_row)
            if progress_bar is not None:
                progress_bar.update()
        meta["fobj"].close()
    output_fobj.close()
    if progress_bar is not None:
        progress_bar.close()
