# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg292::os.mkdir+os.stat
# name: os_primitive
# summary: Uses os.mkdir, os.stat across 2 repos
# anchor_symbols: ['os.mkdir', 'os.stat']
# observed in 2 repos: ['lazyprogrammer__machine_learning_examples', 'tflearn__tflearn']...

# --- from tflearn__tflearn::tflearn/datasets/oxflower17.py::build_class_directories ---
def build_class_directories(dir):
    dir_id = 0
    class_dir = os.path.join(dir, str(dir_id))
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    for i in range(1, 1361):
        fname = "image_" + ("%.4i" % i) + ".jpg"
        os.rename(os.path.join(dir, fname), os.path.join(class_dir, fname))
        if i % 80 == 0 and dir_id < 16:
            dir_id += 1
            class_dir = os.path.join(dir, str(dir_id))
            os.mkdir(class_dir)

# --- from lazyprogrammer__machine_learning_examples::tensorflow/input_data.py::maybe_download ---
def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath

# --- from tflearn__tflearn::tflearn/datasets/svhn.py::maybe_download ---
def maybe_download(filename, work_directory):
    """Download the data from Stanford's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print('Downloading SVHN Dataset...')
        filepath, _ = urllib.request.urlretrieve(URL_BASE + filename,filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath
