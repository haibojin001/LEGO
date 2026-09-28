# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg355::numpy.full+numpy.sqrt+typing.cast
# name: numpy_typing_primitive
# summary: Uses numpy.full, numpy.sqrt, typing.cast across 2 repos
# anchor_symbols: ['numpy.full', 'numpy.sqrt', 'typing.cast']
# observed in 2 repos: ['awslabs__gluonts', 'capitalone__DataProfiler']...

# --- from capitalone__DataProfiler::dataprofiler/profilers/profile_builder.py::StructuredProfiler._get_correlation ---
def _get_correlation(
        self, clean_samples: dict, batch_properties: dict
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix on the cleaned data.

        :param clean_samples: the input cleaned dataset
        :type clean_samples: dict()
        :param batch_properties: mean/std/counts of each batch column necessary
        for correlation computation
        :type batch_properties: dict()
        :return: correlation matrix
        :rtype: pd.DataFrame
        """
        columns = self.options.correlation.columns
        column_ids = list(range(len(self._profile)))
        if columns is not None:
            column_ids = [
                idx for col_name in columns for idx in self._col_name_to_idx[col_name]
            ]
        clean_column_ids = []
        for idx in column_ids:
            data_type = cast(
                ColumnPrimitiveTypeProfileCompiler,
                self._profile[idx].profiles["data_type_profile"],
            ).selected_data_type
            if data_type not in ["int", "float"]:
                clean_samples.pop(idx)
            else:
                clean_column_ids.append(idx)
        data = pd.DataFrame(clean_samples).apply(pd.to_numeric, errors="coerce")
        means = {index: mean for index, mean in enumerate(batch_properties["mean"])}
        data = data.fillna(value=means)
        data = data[clean_column_ids]

        # Update the counts/std if needed (i.e. if null rows or exist)
        if (len(data) != batch_properties["count"]).any():
            adjusted_stds = np.sqrt(
                batch_properties["std"] ** 2
                * (batch_properties["count"] - 1)
                / (len(data) - 1)
            )
            batch_properties["std"] = adjusted_stds
        # Set count key to a single number now that everything's been adjusted
        batch_properties["count"] = len(data)

        # fill correlation matrix with nan initially
        n_cols = len(self._profile)
        corr_mat = np.full((n_cols, n_cols), np.nan)

        # then, fill in the correlations for valid columns
        rows = [[id] for id in clean_column_ids]
        corr_mat[rows, clean_column_ids] = np.corrcoef(data, rowvar=False)

        return corr_mat

# --- from capitalone__DataProfiler::dataprofiler/profilers/profile_builder.py::StructuredProfiler._get_correlation_dependent_properties ---
def _get_correlation_dependent_properties(self, batch: dict = None) -> dict:
        """
        Obtain mean/stddev for calculating correlation.

        By default, it will compute it on all columns in the profiler,
        but if a batch is given, it will compute it only for the columns
        in the batch.

        :param batch: Batch data
        :type batch: dict
        :return: dependent properties
        :rtype: dict
        """
        dependent_properties = {
            "mean": np.full(len(self._profile), np.nan),
            "std": np.full(len(self._profile), np.nan),
            "count": np.full(len(self._profile), np.nan),
        }
        for id in range(len(self._profile)):
            compiler = self._profile[id]
            data_type_compiler = compiler.profiles["data_type_profile"]
            data_type = cast(
                ColumnPrimitiveTypeProfileCompiler, data_type_compiler
            ).selected_data_type
            if data_type in ["int", "float"]:
                data_type_profiler = data_type_compiler._profiles[data_type]
                # Finding dependent values of previous, existing data
                if batch is None:
                    n = data_type_profiler.match_count
                    dependent_properties["mean"][id] = data_type_profiler.mean
                    # Subtract null row count as those aren't included in corr. calc
                    dependent_properties["std"][id] = np.sqrt(
                        data_type_profiler._biased_variance
                        * n
                        / (self.total_samples - self.row_is_null_count - 1)
                    )
                    dependent_properties["count"][id] = n
                # Finding the properties of the batch data if given
                elif id in batch.keys():
                    history = data_type_profiler._batch_history[-1]
                    n = history["match_count"]
                    # Since we impute values, we want the total rows (including nulls)
                    dependent_properties["mean"][id] = history["mean"]
                    dependent_properties["std"][id] = np.sqrt(
                        history["biased_variance"] * n / (n - 1)
                    )
                    dependent_properties["count"][id] = n

        return dependent_properties

# --- from awslabs__gluonts::test/dataset/test_stat.py::DatasetStatisticsExceptions.test_dataset_statistics_exceptions ---
def test_dataset_statistics_exceptions(self) -> None:
        def check_error_message(expected_regex, dataset) -> None:
            with self.assertRaisesRegex(GluonTSDataError, expected_regex):
                calculate_dataset_statistics(dataset)

        check_error_message("Time series dataset is empty!", [])

        check_error_message(
            "Only empty time series found in the dataset!",
            [make_time_series(target=np.random.randint(0, 10, 0))],
        )

        # infinite target
        # check_error_message(
        #     "Target values have to be finite (e.g., not inf, -inf, "
        #     "or None) and cannot exceed single precision floating "
        #     "point range.",
        #     [make_time_series(target=np.full(20, np.inf))]
        # )

        # different number of feat_dynamic_{cat, real}
        check_error_message(
            "Found instances with different number of features in "
            "feat_dynamic_cat, found one with 2 and another with 1.",
            [
                make_time_series(num_feat_dynamic_cat=2),
                make_time_series(num_feat_dynamic_cat=1),
            ],
        )
        check_error_message(
            "Found instances with different number of features in "
            "feat_dynamic_cat, found one with 0 and another with 1.",
            [
                make_time_series(num_feat_dynamic_cat=0),
                make_time_series(num_feat_dynamic_cat=1),
            ],
        )
        check_error_message(
            "feat_dynamic_cat was found for some instances but not others.",
            [
                make_time_series(num_feat_dynamic_cat=1),
                make_time_series(num_feat_dynamic_cat=0),
            ],
        )
        check_error_message(
            "Found instances with different number of features in "
            "feat_dynamic_real, found one with 2 and another with 1.",
            [
                make_time_series(num_feat_dynamic_real=2),
                make_time_series(num_feat_dynamic_real=1),
            ],
        )
        check_error_message(
            "Found instances with different number of features in "
            "feat_dynamic_real, found one with 0 and another with 1.",
            [
                make_time_series(num_feat_dynamic_real=0),
                make_time_series(num_feat_dynamic_real=1),
            ],
        )
        check_error_message(
            "feat_dynamic_real was found for some instances but not others.",
            [
                make_time_series(num_feat_dynamic_real=1),
                make_time_series(num_feat_dynamic_real=0),
            ],
        )

        # infinite feat_dynamic_{cat,real}
        inf_dynamic_feat = np.full((2, len(target)), np.inf)
        check_error_message(
            "Features values have to be finite and cannot exceed single "
            "precision floating point range.",
            [
                ts(
                    start,
                    target,
                    feat_dynamic_cat=inf_dynamic_feat,
                    feat_static_cat=[0, 1],
                )
            ],
        )
        check_error_message(
            "Features values have to be finite and cannot exceed single "
            "precision floating point range.",
            [
                ts(
                    start,
                    target,
                    feat_dynamic_real=inf_dynamic_feat,
                    feat_static_cat=[0, 1],
                )
            ],
        )

        # feat_dynamic_{cat, real} different length from target
        check_error_message(
            "Each feature in feat_dynamic_cat has to have the same length as the "
            "target. Found an instance with feat_dynamic_cat of length 1 and a "
            "target of length 20.",
            [
                ts(
                    start=start,
                    target=target,
                    feat_static_cat=[0, 1],
                    feat_dynamic_cat=np.ones((1, 1)),
                )
            ],
        )
        check_error_message(
            "Each feature in feat_dynamic_real has to have the same length as the "
            "target. Found an instance with feat_dynamic_real of length 1 and a "
            "target of length 20.",
            [
                ts(
                    start=start,
                    target=target,
                    feat_static_cat=[0, 1],
                    feat_dynamic_real=np.ones((1, 1)),
                )
            ],
        )

        # feat_static_{cat, real} different length
        check_error_message(
            "Not all feat_static_cat vectors have the same length 2 != 1.",
            [
                ts(start=start, target=target, feat_static_cat=[0, 1]),
                ts(start=start, target=target, feat_static_cat=[1]),
            ],
        )
        check_error_message(
            "Not all feat_static_real vectors have the same length 2 != 1.",
            [
                ts(start=start, target=target, feat_static_real=[0, 1]),
                ts(start=start, target=target, feat_static_real=[1]),
            ],
        )

        calculate_dataset_statistics(
            # FIXME: the cast below is a hack to make mypy happy
            cast(
                Dataset,
                [
                    make_time_series(num_feat_dynamic_cat=2),
                    make_time_series(num_feat_dynamic_cat=2),
                ],
            )
        )

        calculate_dataset_statistics(
            # FIXME: the cast below is a hack to make mypy happy
            cast(
                Dataset,
                [
                    make_time_series(num_feat_dynamic_cat=0),
                    make_time_series(num_feat_dynamic_cat=0),
                ],
            )
        )
