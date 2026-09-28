# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg792::csv.writer+detect.errorloaderdetector.ErrorsLoaderDetector+tempfile.NamedTemporaryFile
# name: csv_detect_primitive
# summary: Uses csv.writer, detect.errorloaderdetector.ErrorsLoaderDetector, tempfile.NamedTemporaryFile across 3 repos
# anchor_symbols: ['csv.writer', 'detect.errorloaderdetector.ErrorsLoaderDetector', 'tempfile.NamedTemporaryFile']
# observed in 3 repos: ['HoloClean__holoclean', 'probcomp__bayeslite', 'rpy2__rpy2']...

# --- from rpy2__rpy2::rpy2-robjects/src/rpy2/robjects/tests/robjects/test_dataframe.py::test_from_csvfile ---
def test_from_csvfile():
    column_names = ('letter', 'value')
    data = (column_names,
            ('a', 1),
            ('b', 2),
            ('c', 3))
    fh = tempfile.NamedTemporaryFile(mode = "w", delete = False)
    csv_w = csv.writer(fh)
    csv_w.writerows(data)
    fh.close()
    dataf = robjects.DataFrame.from_csvfile(fh.name)
    assert isinstance(dataf, robjects.DataFrame)
    assert column_names == tuple(dataf.names)
    assert dataf.nrow == 3
    assert dataf.ncol == 2

# --- from HoloClean__holoclean::tests/detection/test_errorsloaderdetector.py::test_errors_loader_invalid_csv_file ---
def test_errors_loader_invalid_csv_file():
    tmp_file = NamedTemporaryFile(delete=False)
    with open(tmp_file.name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['_tid_', 'invalid_column'])  # Header.
        csv_writer.writerow([1, 'val1'])

    with pytest.raises(Exception) as invalid_file_error:
        errors_loader_detector = ErrorsLoaderDetector(fpath=tmp_file.name)

    assert 'The loaded errors table does not match the expected schema' in str(invalid_file_error.value)

# --- from HoloClean__holoclean::tests/detection/test_errorsloaderdetector.py::test_errors_loader_valid_csv_file ---
def test_errors_loader_valid_csv_file():
    tmp_file = NamedTemporaryFile(delete=False)
    with open(tmp_file.name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['_tid_', 'attribute'])  # Header.
        csv_writer.writerow([1, 'attr1'])
        csv_writer.writerow([1, 'attr2'])
        csv_writer.writerow([2, 'attr1'])
        csv_writer.writerow([3, 'attr2'])
    errors_loader_detector = ErrorsLoaderDetector(fpath=tmp_file.name)
    errors_df = errors_loader_detector.detect_noisy_cells()

    assert errors_df is not None
    assert errors_df.columns.tolist() == ['_tid_', 'attribute']
    assert len(errors_df) == 4

# --- from probcomp__bayeslite::src/backends/loom_backend.py::LoomBackend._data_to_csv ---
def _data_to_csv(self, bdb, headers, data):
        # TODO: Fix the use of delete=False so loom doesn't litter
        #   the filesystem with the files used to communicate with loom.
        with tempfile.NamedTemporaryFile(delete=False) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=CSV_DELIMITER)
            csv_writer.writerow(headers)
            for row in data:
                processed_row = []
                for elem in row:
                    if elem is None:
                        processed_row.append('')
                    elif isinstance(elem, unicode):
                        processed_row.append(elem.encode('ascii', 'ignore'))
                    else:
                        processed_row.append(elem)
                csv_writer.writerow(processed_row)
        return csv_file
