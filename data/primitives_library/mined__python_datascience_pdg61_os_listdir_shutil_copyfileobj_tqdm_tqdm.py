# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg61::os.listdir+shutil.copyfileobj+tqdm.tqdm
# name: os_shutil_primitive
# summary: Uses os.listdir, shutil.copyfileobj, tqdm.tqdm, zipfile.ZipFile across 4 repos
# anchor_symbols: ['os.listdir', 'shutil.copyfileobj', 'tqdm.tqdm', 'zipfile.ZipFile']
# observed in 4 repos: ['ahmetozlu__tensorflow_object_counting_api', 'fraunhoferportugal__tsfel', 'recommenders-team__recommenders', 'shashankvemuri__Finance']...

# --- from shashankvemuri__Finance::stock_analysis/sp500_cot_sentiment_analysis.py::download_and_extract_cot_file ---
def download_and_extract_cot_file(url, file_name):
    # Download and extract COT file
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    with zipfile.ZipFile(file_name) as zf:
        zf.extractall()

# --- from recommenders-team__recommenders::recommenders/datasets/movielens.py::extract_movielens ---
def extract_movielens(size, rating_path, item_path, zip_path):
    """Extract MovieLens rating and item datafiles from the MovieLens raw zip file.

    To extract all files instead of just rating and item datafiles,
    use ZipFile's extractall(path) instead.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        rating_path (str): Destination path for rating datafile
        item_path (str): Destination path for item datafile
        zip_path (str): zipfile path
    """
    with ZipFile(zip_path, "r") as z:
        with z.open(DATA_FORMAT[size].path) as zf, open(rating_path, "wb") as f:
            shutil.copyfileobj(zf, f)
        with z.open(DATA_FORMAT[size].item_path) as zf, open(item_path, "wb") as f:
            shutil.copyfileobj(zf, f)

# --- from fraunhoferportugal__tsfel::tests/datasets/empirical1000_dataset.py::Empirical1000Dataset.__init__ ---
def __init__(self, use_cache=True, extract_features=False):
        self.use_cache = use_cache

        self.raw = {}
        self.features = {}  # An add-on to this class will contain a featurized data representation.

        self.__dataset_url = "https://figshare.com/ndownloader/articles/5436136/versions/10"
        self.cache_folder_path = data_path
        self.__empirical1000_folder_path = os.path.join(
            self.cache_folder_path,
            "Empirical1000",
        )

        if not os.path.exists(self.cache_folder_path) or not os.listdir(self.cache_folder_path) or not use_cache:
            print("Cache folder is empty. Downloading the Empirical1000 dataset...")
            Path(os.path.join(self.cache_folder_path, "Empirical1000")).mkdir(
                parents=True,
                exist_ok=True,
            )
            self.__download_dataset()

        self.__data_files = sio.loadmat(
            os.path.join(self.__empirical1000_folder_path, "INP_1000ts.mat"),
        )
        self.metadata = pd.read_csv(
            os.path.join(self.__empirical1000_folder_path, "hctsa_timeseries-info.csv"),
        )

        for i, name in enumerate(self.metadata["Name"]):
            self.raw[i] = EmpiricalTimeSeries(
                name,
                self.__data_files["timeSeriesData"][0][i].flatten(),
            )
            if extract_features:
                tqdm_iterator = tqdm(
                    total=len(self.metadata["Name"]),
                    desc="Extracting features.",
                )
                tqdm_iterator.update(1)

# --- from fraunhoferportugal__tsfel::tests/datasets/empirical1000_dataset.py::Empirical1000Dataset.__download_dataset ---
def __download_dataset(self):
        try:
            # Send a request to the dataset URL with allow_redirects=False to handle redirection
            response = requests.get(
                self.__dataset_url,
                stream=True,
                allow_redirects=False,
            )

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0)) or None
                filename = self.__extract_filename(response)

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading Empirical1000 dataset",
                    dynamic_ncols=True,
                ) as pbar:
                    with open(
                        os.path.join(self.cache_folder_path, filename),
                        "wb",
                    ) as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                        pbar.update(os.path.getsize(out_file.name))

                zip_file_path = os.path.join(self.cache_folder_path, filename)
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(self.__empirical1000_folder_path)

                # A sanitizing routine that deletes non-relevant files.
                os.remove(zip_file_path)
                extracted_files = os.listdir(self.__empirical1000_folder_path)
                timeseries_files = [
                    file for file in extracted_files if "timeseries-info" in file.lower() or "inp" in file.lower()
                ]

                for file in extracted_files:
                    if file not in timeseries_files:
                        os.remove(os.path.join(self.__empirical1000_folder_path, file))

                print(f"Dataset downloaded and saved to: {self.cache_folder_path}")
            else:
                print(
                    f"Failed to download dataset. Status code: {response.status_code}",
                )

        except Exception as e:
            print(f"An error occurred: {e}")

# --- from ahmetozlu__tensorflow_object_counting_api::mask_rcnn_counting_api/coco.py::CocoDataset.auto_download ---
def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)
