# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg429::random.sample+typing.cast
# name: random_typing_primitive
# summary: Uses random.sample, typing.cast across 3 repos
# anchor_symbols: ['random.sample', 'typing.cast']
# observed in 3 repos: ['capitalone__DataProfiler', 'cleanlab__cleanvision', 'deepchecks__deepchecks']...

# --- from capitalone__DataProfiler::dataprofiler/profilers/profile_builder.py::UnstructuredProfiler.__add__ ---
def __add__(  # type: ignore[override]
        self, other: UnstructuredProfiler
    ) -> UnstructuredProfiler:
        """
        Merge two Unstructured profiles together overriding the `+` operator.

        :param other: unstructured profile being added to this one.
        :type other: UnstructuredProfiler
        :return: merger of the two profiles
        :rtype: UnstructuredProfiler
        """
        merged_profile = cast(UnstructuredProfiler, super().__add__(other))

        # unstruct specific property merging
        merged_profile._empty_line_count = (
            self._empty_line_count + other._empty_line_count
        )
        merged_profile.memory_size = self.memory_size + other.memory_size
        samples = list(dict.fromkeys(self.sample + other.sample))
        merged_profile.sample = random.sample(list(samples), min(len(samples), 5))

        # merge profiles
        merged_profile._profile = self._profile + other._profile

        return merged_profile

# --- from deepchecks__deepchecks::deepchecks/nlp/checks/data_integrity/special_characters.py::SpecialCharacters.run_logic ---
def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        dataset = t.cast(TextData, dataset)

        if dataset.n_samples == 0:
            raise DeepchecksValueError('Dataset cannot be empty')

        samples_per_special_char = {}
        percent_special_chars_in_sample = {}

        for idx, sample in zip(dataset.get_original_text_indexes(), dataset.text):
            if pd.isna(sample):
                continue
            if len(sample) > self.max_chars_to_review_per_sample:
                sample = random.sample(sample, self.max_chars_to_review_per_sample)
            if len(sample) == 0:
                percent_special_chars_in_sample[idx] = 0
                continue
            special_chars_in_sample = [char for char in sample if char in self.special_characters_deny_list]
            percent_special_chars_in_sample[idx] = len(special_chars_in_sample) / len(sample)
            for char in frozenset(special_chars_in_sample):
                base = samples_per_special_char[char] if char in samples_per_special_char else []
                samples_per_special_char[char] = base + [idx]

        percents_arr = np.asarray(list(percent_special_chars_in_sample.values()))
        percent_of_samples_with_special_chars = len(percents_arr[percents_arr > 0]) / dataset.n_samples
        percent_special_chars_in_sample = pd.Series(percent_special_chars_in_sample).sort_values(ascending=False)
        samples_per_special_char = dict(sorted(samples_per_special_char.items(), key=lambda x: -len(x[1])))
        result_value = {
            'samples_per_special_char': samples_per_special_char,
            'percent_of_samples_with_special_chars': percent_of_samples_with_special_chars,
            'percent_special_chars_per_sample': pd.Series(percent_special_chars_in_sample),
        }

        if context.with_display is False or len(samples_per_special_char) == 0:
            return CheckResult(value=result_value)

        display_table = pd.DataFrame(columns=['Sample ID', '% of Special Characters',
                                              'Special Characters', 'Text'])
        for idx, value in percent_special_chars_in_sample[:self.max_samples_to_show].items():
            text_sample = dataset.get_sample_at_original_index(idx)
            special_chars = Counter(char for char in text_sample if char in self.special_characters_deny_list)
            special_chars = [x[0] for x in special_chars.most_common()[:self.max_special_chars_to_show]]
            display_table.loc[len(display_table)] = \
                [idx, value, special_chars, text_sample[:self.max_text_length_for_display]]

        return CheckResult(
            value=result_value,
            display=[
                f'<b>{format_percent(percent_of_samples_with_special_chars)}</b> of samples contain special characters',
                f'List of ignored special characters: {list(self.special_characters_allow_list)}',
                hide_index_for_display(display_table)
            ]
        )

# --- from cleanlab__cleanvision::src/cleanvision/imagelab.py::Imagelab.visualize ---
def visualize(
        self,
        image_files: Optional[List[str]] = None,
        indices: Optional[List[str | int]] = None,
        issue_types: Optional[List[str]] = None,
        num_images: int = 4,
        cell_size: Tuple[int, int] = (2, 2),
        show_id: bool = False,
    ) -> None:
        """Show specific images.

        Can be used for visualizing either:
        1. Particular images with paths given in `image_files`.
        2. Images representing top-most severe instances of given `issue_types` detected the dataset.
        3. If no `image_files` or `issue_types` are given, random images will be shown from the dataset.

        If `image_files` is given, this overrides the argument `issue_types`.

        Parameters
        ----------

        image_files : List[str], optional
            List of filepaths for images to visualize.

        indices: List[str|int], optional
            List of indices of images in the dataset to visualize.
            If the dataset is a local data_path, the indices are filepaths, which is also the index in `imagelab.issues` dataframe.
            If the dataset is a huggingface or torchvision dataset, indices are of type int and corresponding to the indices in the dataset object.


        issue_types: List[str], optional
            List of issue types to visualize. For each type of issue, will show a few images representing the top-most severe instances of this issue in the dataset.

        num_images : int, optional
            Number of images to visualize from the dataset.
            These images are randomly selected if `issue_types` is ``None``.
            If `issue_types` is given, then this is the number of images for each issue type to visualize
            (images representing top-most severe instances of this issue will be shown).
            If `image_files` is given, this argument is ignored.

        cell_size : Tuple[int, int], optional
            Dimensions controlling the size of each image in the depicted image grid.

        Examples
        --------

        To visualize random images from the dataset

        .. code-block:: python

            imagelab.visualize()

        .. code-block:: python

            imagelab.visualize(num_images=8)

        To visualize specific images from the dataset

        .. code-block:: python

            image_files = ["./dataset/cat.png", "./dataset/dog.png", "./dataset/mouse.png"]
            imagelab.visualize(image_files=image_files)

        To visualize top examples of specific issue types from the dataset

        .. code-block:: python

            issue_types = ["dark", "odd_aspect_ratio"]
            imagelab.visualize(issue_types=issue_types)

        """
        if issue_types is not None:
            if len(issue_types) == 0:
                raise ValueError("issue_types list is empty")
            for issue_type in issue_types:
                self._visualize(issue_type, num_images, cell_size, show_id)
        elif image_files is not None:
            if len(image_files) == 0:
                raise ValueError("image_files list is empty.")
            images: List[Image.Image] = [Image.open(path) for path in image_files]
            title_info = {"path": [path.split("/")[-1] for path in image_files]}
            VizManager.individual_images(
                images,
                title_info,
                ncols=self._config["visualize_num_images_per_row"],
                cell_size=cell_size,
            )
        elif indices:
            images = [cast(Image.Image, self._dataset[i]) for i in indices]
            title_info = {"name": [self._dataset.get_name(i) for i in indices]}
            VizManager.individual_images(
                images,
                title_info,
                ncols=self._config["visualize_num_images_per_row"],
                cell_size=cell_size,
            )
        else:
            print("Sample images from the dataset")

            if image_files is None:
                image_indices = random.sample(
                    self._dataset.index, min(num_images, len(self._dataset))
                )
                images = [cast(Image.Image, self._dataset[i]) for i in image_indices]
                title_info = {
                    "name": [self._dataset.get_name(i) for i in image_indices]
                }
                VizManager.individual_images(
                    images,
                    title_info,
                    ncols=self._config["visualize_num_images_per_row"],
                    cell_size=cell_size,
                )
