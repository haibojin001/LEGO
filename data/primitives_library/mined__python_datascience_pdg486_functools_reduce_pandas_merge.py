# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg486::functools.reduce+pandas.merge
# name: functools_pandas_primitive
# summary: Uses functools.reduce, pandas.merge across 2 repos
# anchor_symbols: ['functools.reduce', 'pandas.merge']
# observed in 2 repos: ['OML-Team__open-metric-learning', 'stellargraph__stellargraph']...

# --- from stellargraph__stellargraph::demos/community_detection/utils.py::load_features ---
def load_features(input_data):
    # Summarise features by terrorist group
    dt_collect = input_data[
        ["eventid", "nperps", "success", "suicide", "nkill", "nwound", "gname"]
    ]
    dt_collect.fillna(0, inplace=True)
    dt_collect.nperps[dt_collect.nperps < 0] = 0

    summarize_by_gname = (
        dt_collect.groupby("gname")
        .agg(
            {
                "eventid": "count",
                "nperps": "sum",
                "nkill": "sum",
                "nwound": "sum",
                "success": "sum",
            }
        )
        .reset_index()
    )
    summarize_by_gname.columns = [
        "gname",
        "n_attacks",
        "n_nperp",
        "n_nkil",
        "n_nwound",
        "n_success",
    ]
    summarize_by_gname["success_ratio"] = (
        summarize_by_gname["n_success"] / summarize_by_gname["n_attacks"]
    )
    summarize_by_gname.drop(["n_success"], axis=1, inplace=True)

    # Collect counts of each attack type
    dt_collect = input_data[["gname", "attacktype1_txt"]]
    gname_attacktypes = (
        dt_collect.groupby(["gname", "attacktype1_txt"])["attacktype1_txt"]
        .count()
        .to_frame()
    )
    gname_attacktypes.columns = ["attacktype_count"]
    gname_attacktypes.reset_index(inplace=True)
    gname_attacktypes_wide = gname_attacktypes.pivot(
        index="gname", columns="attacktype1_txt", values="attacktype_count"
    )
    gname_attacktypes_wide.fillna(0, inplace=True)
    gname_attacktypes_wide.drop(["Unknown"], axis=1, inplace=True)

    # Collect counts of each target type
    dt_collect = input_data[["gname", "targtype1_txt"]]
    gname_targtypes = (
        dt_collect.groupby(["gname", "targtype1_txt"])["targtype1_txt"]
        .count()
        .to_frame()
    )
    gname_targtypes.columns = ["targtype_count"]
    gname_targtypes.reset_index(inplace=True)
    gname_targtypes_wide = gname_targtypes.pivot(
        index="gname", columns="targtype1_txt", values="targtype_count"
    )
    gname_targtypes_wide.fillna(0, inplace=True)
    gname_targtypes_wide.drop(["Unknown"], axis=1, inplace=True)

    # Combine all features
    data_frames = [summarize_by_gname, gname_attacktypes_wide, gname_targtypes_wide]
    gnames_features = reduce(
        lambda left, right: pd.merge(left, right, on=["gname"], how="outer"),
        data_frames,
    )
    return gnames_features

# --- from OML-Team__open-metric-learning::pipelines/datasets_converters/convert_cub.py::build_cub_df ---
def build_cub_df(dataset_root: Path, no_bboxes: bool) -> pd.DataFrame:
    dataset_root = Path(dataset_root)

    images_txt = dataset_root / "images.txt"
    train_test_split = dataset_root / "train_test_split.txt"
    bounding_boxes = dataset_root / "bounding_boxes.txt"
    image_class_labels = dataset_root / "image_class_labels.txt"

    for file in [images_txt, train_test_split, bounding_boxes, image_class_labels]:
        assert file.is_file(), f"File {file} does not exist."

    with open(images_txt, "r") as f:
        images = f.read()
        images = pd.read_csv(io.StringIO(images), delim_whitespace=True, header=None, names=["image_id", "image_name"])

    with open(train_test_split, "r") as f:
        split = f.read()
        split = pd.read_csv(
            io.StringIO(split), delim_whitespace=True, header=None, names=["image_id", "is_training_image"]
        )

    with open(bounding_boxes, "r") as f:
        bbs = f.read()
        bbs = pd.read_csv(
            io.StringIO(bbs), delim_whitespace=True, header=None, names=["image_id", "x", "y", "width", "height"]
        )

    with open(image_class_labels, "r") as f:
        class_labels = f.read()
        class_labels = pd.read_csv(
            io.StringIO(class_labels), delim_whitespace=True, header=None, names=["image_id", "class_id"]
        )

    df = ft.reduce(lambda left, right: pd.merge(left, right, on="image_id"), [images, bbs, class_labels, split])

    df["x_1"] = df["x"].apply(int)  # left
    df["x_2"] = (df["x"] + df["width"]).apply(int)  # right
    df["y_2"] = (df["y"] + df["height"]).apply(int)  # bot
    df["y_1"] = df["y"].apply(int)  # top
    df["path"] = df["image_name"].apply(lambda x: Path("images") / x)

    df["split"] = "train"
    df["split"][df["is_training_image"] == 0] = "validation"

    df["is_query"] = None
    df["is_gallery"] = None
    df["is_query"][df["split"] == "validation"] = True
    df["is_gallery"][df["split"] == "validation"] = True

    df = df.rename(columns={"class_id": "label"})

    cols_to_pick = ["label", "path", "split", "is_query", "is_gallery"]
    if not no_bboxes:
        cols_to_pick.extend(["x_1", "x_2", "y_1", "y_2"])
    df = df[cols_to_pick]

    df = df.rename(
        columns={
            "label": LABELS_COLUMN,
            "path": PATHS_COLUMN,
            "split": SPLIT_COLUMN,
            "is_query": IS_QUERY_COLUMN,
            "is_gallery": IS_GALLERY_COLUMN,
            "x_1": X1_COLUMN,
            "x_2": X2_COLUMN,
            "y_1": Y1_COLUMN,
            "y_2": Y2_COLUMN,
        }
    )

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)
    return df
