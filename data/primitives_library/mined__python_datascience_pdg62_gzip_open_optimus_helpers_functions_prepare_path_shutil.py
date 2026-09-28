# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg62::gzip.open+optimus.helpers.functions.prepare_path+shutil.copyfileobj
# name: gzip_optimus_primitive
# summary: Uses gzip.open, optimus.helpers.functions.prepare_path, shutil.copyfileobj across 3 repos
# anchor_symbols: ['gzip.open', 'optimus.helpers.functions.prepare_path', 'shutil.copyfileobj']
# observed in 3 repos: ['hi-primus__optimus', 'insitro__redun', 'recommenders-team__recommenders']...

# --- from recommenders-team__recommenders::recommenders/datasets/amazon_reviews.py::_extract_reviews ---
def _extract_reviews(file_path, zip_path):
    """Extract Amazon reviews and meta datafiles from the raw zip files.

    To extract all files,
    use ZipFile's extractall(path) instead.

    Args:
        file_path (str): Destination path for datafile
        zip_path (str): zipfile path
    """
    with gzip.open(zip_path + ".gz", "rb") as zf, open(file_path, "wb") as f:
        shutil.copyfileobj(zf, f)

# --- from hi-primus__optimus::optimus/engines/polars/io/extract.py::Extract.gz ---
def gz(path, *args, **kwargs):
        """
        Loads a dataframe from a gz file.
        :param path: path or location of the file. Must be string dataType
        :param args: custom argument to be passed to the internal function
        :param kwargs: custom keyword arguments to be passed to the internal function
        :return: Spark Dataframe
        """
        file, file_name = prepare_path(path, "gz")

        import gzip
        import shutil
        with gzip.open(file, 'rb') as f_in:
            print(f_in)
            with open('file.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(file, file_name)

# --- from hi-primus__optimus::optimus/engines/pandas/io/extract.py::Extract.gz ---
def gz(path, *args, **kwargs):
        """
        Loads a dataframe from a gz file.
        :param path: path or location of the file. Must be string dataType
        :param args: custom argument to be passed to the internal function
        :param kwargs: custom keyword arguments to be passed to the internal function
        :return: Spark Dataframe
        """
        file, file_name = prepare_path(path, "gz")

        import gzip
        import shutil
        with gzip.open(file, 'rb') as f_in:
            print(f_in)
            with open('file.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(file, file_name)

# --- from insitro__redun::examples/06_bioinfo_batch/workflow.py::download_genome_ref ---
def download_genome_ref(genome_ref_src: File, ref_dir: str, skip_if_exists: bool = True) -> File:
    """
    Download reference genome to our own directory/bucket.
    """
    dest_path = os.path.join(ref_dir, genome_ref_src.basename())
    if genome_ref_src.path.endswith(".gz"):
        # Unzip genome if it is zipped.
        dest_file = File(dest_path[: -len(".gz")])
        if not skip_if_exists or not dest_file.exists():
            with dest_file.open("wb") as outfile:
                with genome_ref_src.open("rb") as infile:
                    with gzip.open(infile, "rb") as infile2:
                        shutil.copyfileobj(infile2, outfile)
    else:
        # Copy genome as is.
        dest_file = File(dest_path)
        genome_ref_src.copy_to(dest_file, skip_if_exists=skip_if_exists)

    return dest_file
