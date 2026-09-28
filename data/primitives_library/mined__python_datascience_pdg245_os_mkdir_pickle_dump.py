# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg245::os.mkdir+pickle.dump
# name: os_pickle_primitive
# summary: Uses os.mkdir, pickle.dump across 2 repos
# anchor_symbols: ['os.mkdir', 'pickle.dump']
# observed in 2 repos: ['cleanlab__cleanlab', 'cleanlab__cleanvision']...

# --- from cleanlab__cleanlab::cleanlab/datalab/internal/serialize.py::_Serializer.serialize ---
def serialize(cls, path: str, datalab: Datalab, force: bool) -> None:
        """Serializes the datalab object to disk.

        Parameters
        ----------
        path : str
            Path to save the datalab object to.

        datalab : Datalab
            The datalab object to save.

        force : bool
            If True, will overwrite existing files at the specified path.
        """
        path_exists = os.path.exists(path)
        if not path_exists:
            os.mkdir(path)
        else:
            if not force:
                raise FileExistsError("Please specify a new path or set force=True")
            print(f"WARNING: Existing files will be overwritten by newly saved files at: {path}")

        # Save the datalab object to disk.
        with open(os.path.join(path, OBJECT_FILENAME), "wb") as f:
            pickle.dump(datalab, f)

        # Save the issues to disk. Use placeholder method for now.
        cls._save_data_issues(path=path, datalab=datalab)

        # Save the dataset to disk
        cls._save_data(path=path, datalab=datalab)

# --- from cleanlab__cleanvision::src/cleanvision/utils/serialize.py::Serializer.serialize ---
def serialize(cls, path: str, imagelab: Imagelab, force: bool) -> None:
        """Serializes the imagelab object to disk.

        Parameters
        ----------
        path : str
            Path to save the imagelab object to.

        imagelab : Imagelab
            The imagelab object to save.

        force : bool
            If True, will overwrite existing files at the specified path.

        Raises
        ------
        FileExistsError
            If `force` is set to False, and an existing path is specified for saving Imagelab instance.

        """
        path_exists = os.path.exists(path)
        if not path_exists:
            os.mkdir(path)
        else:
            if not force:
                raise FileExistsError("Please specify a new path or set force=True")
            print(
                f"WARNING: Existing files will be overwritten by newly saved files at: {path}"
            )

        # Save the issues to disk.
        cls._save_issues(path=path, imagelab=imagelab)

        # clear issues and issue_summary
        imagelab_copy = deepcopy(imagelab)
        imagelab_copy.issues = None
        imagelab_copy.issue_summary = None

        # Save the imagelab object to disk.
        with open(os.path.join(path, OBJECT_FILENAME), "wb") as f:
            pickle.dump(imagelab_copy, f)

        print(f"Saved Imagelab to folder: {path}")
        print(
            "The data path and dataset must be not be changed to maintain consistent state when loading this Imagelab"
        )
