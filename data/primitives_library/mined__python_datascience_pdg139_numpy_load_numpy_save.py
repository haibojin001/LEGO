# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg139::numpy.load+numpy.save
# name: numpy_primitive
# summary: Uses numpy.load, numpy.save across 6 repos
# anchor_symbols: ['numpy.load', 'numpy.save']
# observed in 6 repos: ['cleanlab__cleanlab', 'deepchecks__deepchecks', 'jasmcaus__caer', 'lazyprogrammer__machine_learning_examples', 'microsoft__nni']...

# --- from microsoft__nni::examples/trials/kaggle-tgs-salt/predict.py::ensemble_np ---
def ensemble_np(args, np_files, save_np=None):
    preds = []
    for np_file in np_files:
        pred = np.load(np_file)
        print(np_file, pred.shape)
        preds.append(pred)

    y_pred_test = generate_preds(np.mean(preds, 0), (settings.ORIG_H, settings.ORIG_W), args.pad_mode)

    if save_np is not None:
        np.save(save_np, np.mean(preds, 0))

    meta = get_test_loader(args.batch_size, index=0, dev_mode=False, pad_mode=args.pad_mode).meta

    submission = create_submission(meta, y_pred_test)
    submission.to_csv(args.sub_file, index=None, encoding='utf-8')

# --- from lazyprogrammer__machine_learning_examples::cnn_class/cifar.py::getImageData ---
def getImageData():
    N = 50000
    savedXpath = '../large_files/cifar10/train_all.npy'
    if not os.path.exists(savedXpath):
        X = np.zeros((N, 3, 32, 32))
        for i in xrange(N):
            im = Image.open("../large_files/cifar10/train/%s.png" % (i + 1))
            X[i] = image2array(im)
            if i % 1000 == 0:
                print i
        np.save(savedXpath, X.astype(np.uint8))
    else:
        X = np.load(savedXpath)
    X = X.astype(np.float32) / 255.0

    # load labels
    Y = np.zeros(N)
    df = pd.read_csv('../large_files/cifar10/trainLabels.csv')
    S = df['label'].tolist()
    idx = 0
    label2idx = {}
    i = 0
    for s in S:
        if s not in label2idx:
            label2idx[s] = idx
            idx += 1
        Y[i] = label2idx[s]
        i += 1
    print "done loading data"
    X, Y = shuffle(X, Y)
    return X[:30000], Y[:30000]

# --- from pykale__pykale::tests/prepdata/test_image_transform.py::test_prepare_image_tensor_hw1_to_hw3 ---
def test_prepare_image_tensor_hw1_to_hw3(tmp_path):
    # Simulate (H, W, 1) image
    np_img = np.random.randint(0, 255, size=(32, 32, 1), dtype=np.uint8)
    img_path = tmp_path / "hw1.png"
    # PIL expects at least 2D, save as grayscale
    img = Image.fromarray(np_img.squeeze())
    img.save(img_path)
    # Re-load as (H, W, 1) and save as .npy for direct skimage.io.imread
    np.save(str(tmp_path / "hw1.npy"), np_img)
    # Use prepare_image_tensor (simulate read as H, W, 1)
    img_loaded = np.load(str(tmp_path / "hw1.npy"))
    # Monkeypatch skimage.io.imread to return (H, W, 1)
    import skimage.io

    orig_imread = skimage.io.imread
    skimage.io.imread = lambda path: img_loaded
    try:
        out_tensor = prepare_image_tensor("dummy_path", resize_dim=(224, 224), channels=3)
        assert out_tensor.shape == (3, 224, 224)
    finally:
        skimage.io.imread = orig_imread

# --- from cleanlab__cleanlab::tests/test_segmentation.py::test_results_are_consistent_with_batch_size ---
def test_results_are_consistent_with_batch_size(tmp_path: Path):
    """
    Test that find_label_issues works with large memmap arrays and different batch sizes
    """

    # Create dummy versions of pred_probs and labels
    # write to the pytest tmp_path so that the files are deleted after the test
    pred_probs_file = tmp_path / "pred_probs.npy"
    labels_file = tmp_path / "labels.npy"
    np.save(pred_probs_file, np.random.rand(100, 2, 5, 5))
    np.save(labels_file, np.random.randint(0, 2, (100, 5, 5)))

    # Load the numpy arrays from disk
    pred_probs = np.load(pred_probs_file, mmap_mode="r")
    pred_labels = np.load(labels_file, mmap_mode="r")

    # Test with different batch sizes
    batch_sizes = [1, 50, 100]
    issues_list = []
    for batch_size in batch_sizes:
        issues = find_label_issues(pred_labels, pred_probs, n_jobs=None, batch_size=batch_size)
        issues_list.append(issues)

    # Verify that the results are identical regardless of the batch size
    for i in range(len(batch_sizes) - 1):
        assert np.array_equal(issues_list[i], issues_list[i + 1])

# --- from deepchecks__deepchecks::deepchecks/utils/builtin_datasets_utils.py::read_and_save_data ---
def read_and_save_data(assets_dir, file_name, url_to_file, file_type='csv', to_numpy=False, include_index=True):
    """If the file exist reads it from the assets' directory, otherwise reads it from the url and saves it."""
    os.makedirs(assets_dir, exist_ok=True)
    if (assets_dir / file_name).exists():
        if file_type == 'csv':
            data = pd.read_csv(assets_dir / file_name, index_col=0 if include_index else None)
        elif file_type == 'npy':
            data = np.load(assets_dir / file_name)
        elif file_type == 'json':
            with open(assets_dir / file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError('file_type must be either "csv" or "npy"')
    else:
        if file_type == 'csv':
            data = pd.read_csv(url_to_file, index_col=0 if include_index else None)
            data.to_csv(assets_dir / file_name)
        elif file_type == 'npy':
            data = np.load(BytesIO(requests.get(url_to_file).content))
            np.save(assets_dir / file_name, data)
        elif file_type == 'json':
            data = json.loads(requests.get(url_to_file).content)
            with open(assets_dir / file_name, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        else:
            raise ValueError('file_type must be either "csv" or "npy"')

    if to_numpy and (file_type in {'csv', 'npy'}):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            raise ValueError(f'Unknown data type - {type(data)}. Must be either pandas.DataFrame or numpy.ndarray')
    elif to_numpy:
        raise ValueError(f'Cannot convert {file_type} to numpy array')
    return data

# --- from jasmcaus__caer::caer/preprocess.py::preprocess_from_dir ---
def preprocess_from_dir(
        DIR : str, 
        classes : Optional[List[str]] = None, 
        IMG_SIZE : Optional[Tuple[int,int]] = None, 
        channels : int = 3, 
        isShuffle : bool = True, 
        save_data : bool = False, 
        destination_filename : Optional[str] = None, 
        verbose : bool = True
    ):
    """
    Reads Images in base directory DIR using ``classes`` (computed from sub directories)

    Arguments:
        DIR (str): Base directory 
        classes (list): A list of folder names within `DIR`.
        IMG_SIZE (tuple): Image Size tuple of size 2 (width, height)
        channels (int): Number of channels each image will be processed to (default: 3)
        isShuffle (bool): Shuffle the training set
        save_data (bool): If True, saves the training set as a .npy or .npz file based on destination_filename
        destination_filename (Optional[str]): if save_data is True, the train set will be saved as the filename specified
        verbose (bool): Displays the progress to the terminal as preprocessing continues. Default = True
    
    Returns
        data: Image Pixel Values with corresponding labels (float32)
        Saves the above variables as .npy files if `save_data = True`
    """
    data = [] 

    if not os.path.exists(DIR):
        raise ValueError("The specified directory does not exist")

    if IMG_SIZE is None:
        raise ValueError("IMG_SIZE must be specified")

    if not isinstance(IMG_SIZE, tuple) or len(IMG_SIZE) != 2:
        raise ValueError("IMG_SIZE must be a tuple of size 2 (width,height)")

    if not isinstance(save_data, bool):
        raise ValueError("save_data must be a boolean (True/False)")

    if not isinstance(classes, list):
        raise ValueError("`classes` must be a list")

    if save_data:
        if destination_filename is None:
            raise ValueError("Specify a destination file name")

        elif not (".npy" in destination_filename or ".npz" in destination_filename):
            raise ValueError("Specify the correct numpy destination file extension (.npy or .npz)")
    
    if not save_data and destination_filename is not None:
        destination_filename = None

    # Loading from Numpy Files
    if destination_filename is not None and os.path.exists(destination_filename):
        print("[INFO] Loading from Numpy Files")
        since = time.time()
        data = np.load(destination_filename, allow_pickle=True)
        end = time.time()
        took = end - since
        print("----------------------------------------------")
        print(f"[INFO] Loaded in {took:.0f}s from Numpy Files")

        return data

    # Extracting image data and adding to `data`
    else:
        if destination_filename is not None:
            print(f"[INFO] Could not find {destination_filename}. Generating the training data")
        else:
            print("[INFO] Could not find a file to load from. Generating the training data")
        print("----------------------------------------------")

        # Starting timer
        since_preprocess = time.time()

        for item in classes:
            class_path = join(DIR, item)
            class_label = classes.index(item)
            count = 0 
            tens_list = list_images(class_path, use_fullpath=True, verbose=verbose)

            for image_path in tens_list:
                tens = imread(image_path, target_size=IMG_SIZE, rgb=True)

                if tens is None:
                    continue
                
                # Gray
                if channels == 1:
                    tens = to_gray(tens)

                # Appending to train set
                data.append([tens, class_label])
                count += 1

        # Shuffling the Training Set
        if isShuffle is True:
            random.shuffle(data)

        # Converting to Numpy
        data = np.array(data, dtype=object)

        # Saves the Data set as a .npy file
        if save_data:
            #Converts to Numpy and saves
            if destination_filename.endswith(".npy"):   # type: ignore
                print("[INFO] Saving as .npy file")
            elif destination_filename.endswith(".npz"): # type: ignore
                print("[INFO] Saving as .npz file")
            
            # Saving
            since = time.time()
            np.save(destination_filename, data)
            end = time.time()
            
            time_elapsed = end-since
            minu_elapsed = time_elapsed // 60
            sec_elapsed = time_elapsed % 60
            print(f"[INFO] {destination_filename} saved! Took {minu_elapsed:.0f}m {sec_elapsed:.0f}s")

        #Returns Training Set
        end_preprocess = time.time()
        time_elapsed_preprocess = end_preprocess - since_preprocess
        minu = time_elapsed_preprocess // 60
        sec = time_elapsed_preprocess % 60

        print("----------------------------------------------")
        print(f"[INFO] {len(data)} files preprocessed! Took {minu:.0f}m {sec:.0f}s")

        return data
