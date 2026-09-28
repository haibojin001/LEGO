# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg239::cleanlab.internal.multiannotator_utils.assert_valid_inputs_multiannotator+cleanlab.internal.multiannotator_utils.check_consensus_label_classes+cleanlab.internal.multiannotator_utils.find_best_temp_scaler
# name: cleanlab_numpy_primitive
# summary: Uses cleanlab.internal.multiannotator_utils.assert_valid_inputs_multiannotator, cleanlab.internal.multiannotator_utils.check_consensus_label_classes, cleanlab.internal.multiannotator_utils.find_best_temp_scaler, cleanlab.internal.multiannotator_utils.temp_scale_pred_probs across 2 repos
# anchor_symbols: ['cleanlab.internal.multiannotator_utils.assert_valid_inputs_multiannotator', 'cleanlab.internal.multiannotator_utils.check_consensus_label_classes', 'cleanlab.internal.multiannotator_utils.find_best_temp_scaler', 'cleanlab.internal.multiannotator_utils.temp_scale_pred_probs', 'numpy.array', 'numpy.average', 'numpy.full', 'numpy.isnan', 'numpy.max', 'numpy.mean', 'numpy.sum', 'numpy.where']
# observed in 2 repos: ['cleanlab__cleanlab', 'xorbitsai__xorbits']...

# --- from xorbitsai__xorbits::python/xorbits/_mars/learn/ensemble/_blockwise.py::BlockwiseEnsemblePredict._execute_classifier_predict_proba ---
def _execute_classifier_predict_proba(
        cls, predictions: List[np.ndarray], op: "BlockwiseEnsemblePredict"
    ):
        assert op.voting == "soft"
        return np.average(np.stack(predictions), axis=0)

# --- from cleanlab__cleanlab::cleanlab/multiannotator.py::get_majority_vote_label ---
def get_majority_vote_label(
    labels_multiannotator: Union[pd.DataFrame, np.ndarray],
    pred_probs: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Returns the majority vote label for each example, aggregated from the labels given by multiple annotators.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        2D pandas DataFrame or array of multiple given labels for each example with shape ``(N, M)``,
        where N is the number of examples and M is the number of annotators.
        For more details, labels in the same format expected by the `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    pred_probs : np.ndarray, optional
        An array of shape ``(N, K)`` of model-predicted probabilities, ``P(label=k|x)``.
        For details, predicted probabilities in the same format expected by `~cleanlab.multiannotator.get_label_quality_multiannotator`.
    verbose : bool, optional
        Important warnings and other printed statements may be suppressed if ``verbose`` is set to ``False``.
    Returns
    -------
    consensus_label: np.ndarray
        An array of shape ``(N,)`` with the majority vote label aggregated from all annotators.

        In the event of majority vote ties, ties are broken in the following order:
        using the model ``pred_probs`` (if provided) and selecting the class with highest predicted probability,
        using the empirical class frequencies and selecting the class with highest frequency,
        using an initial annotator quality score and selecting the class that has been labeled by annotators with higher quality,
        and lastly by random selection.
    """

    if isinstance(labels_multiannotator, pd.DataFrame):
        annotator_ids = labels_multiannotator.columns
        labels_multiannotator = (
            labels_multiannotator.replace({pd.NA: np.nan}).astype(float).to_numpy()
        )
    elif isinstance(labels_multiannotator, np.ndarray):
        annotator_ids = None
    else:
        raise ValueError("labels_multiannotator must be either a NumPy array or Pandas DataFrame.")

    if verbose:
        assert_valid_inputs_multiannotator(
            labels_multiannotator, pred_probs, annotator_ids=annotator_ids
        )

    if pred_probs is not None:
        num_classes = pred_probs.shape[1]
    else:
        num_classes = int(np.nanmax(labels_multiannotator) + 1)

    array_idx = np.arange(labels_multiannotator.shape[0])
    label_count = np.zeros((labels_multiannotator.shape[0], num_classes))
    for i in range(labels_multiannotator.shape[1]):
        not_nan_mask = ~np.isnan(labels_multiannotator[:, i])
        # Get the indexes where the label is not missing for the annotator i as int.
        label_index = labels_multiannotator[not_nan_mask, i].astype(int)
        # Increase the counts of those labels by 1.
        label_count[array_idx[not_nan_mask], label_index] += 1

    mode_labels_multiannotator = np.full(label_count.shape, np.nan)
    modes_mask = label_count == np.max(label_count, axis=1).reshape(-1, 1)
    insert_index = np.zeros(modes_mask.shape[0], dtype=int)
    for i in range(modes_mask.shape[1]):
        mode_index = np.where(modes_mask[:, i])[0]
        mode_labels_multiannotator[mode_index, insert_index[mode_index]] = i
        insert_index[mode_index] += 1

    majority_vote_label = np.full(len(labels_multiannotator), np.nan)
    label_mode_count = (~np.isnan(mode_labels_multiannotator)).sum(axis=1)

    # obtaining consensus using annotator majority vote
    mode_count_one_mask = label_mode_count == 1
    majority_vote_label[mode_count_one_mask] = mode_labels_multiannotator[mode_count_one_mask, 0]
    nontied_idx = array_idx[mode_count_one_mask]
    tied_idx = {
        i: label_mode[:count].astype(int)
        for i, label_mode, count in zip(
            array_idx[~mode_count_one_mask],
            mode_labels_multiannotator[~mode_count_one_mask, :],
            label_mode_count[~mode_count_one_mask],
        )
    }

    # tiebreak 1: using pred_probs (if provided)
    if pred_probs is not None and len(tied_idx) > 0:
        for idx, label_mode in tied_idx.copy().items():
            max_pred_probs = np.where(
                pred_probs[idx, label_mode] == np.max(pred_probs[idx, label_mode])
            )[0]
            if len(max_pred_probs) == 1:
                majority_vote_label[idx] = label_mode[max_pred_probs[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[max_pred_probs]

    # tiebreak 2: using empirical class frequencies
    # current tiebreak will select the minority class (to prevent larger class imbalance)
    if len(tied_idx) > 0:
        class_frequencies = label_count.sum(axis=0)
        for idx, label_mode in tied_idx.copy().items():
            min_frequency = np.where(
                class_frequencies[label_mode] == np.min(class_frequencies[label_mode])
            )[0]
            if len(min_frequency) == 1:
                majority_vote_label[idx] = label_mode[min_frequency[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[min_frequency]

    # tiebreak 3: using initial annotator quality scores
    if len(tied_idx) > 0:
        nontied_majority_vote_label = majority_vote_label[nontied_idx]
        nontied_labels_multiannotator = labels_multiannotator[nontied_idx]
        annotator_agreement_with_consensus = np.zeros(nontied_labels_multiannotator.shape[1])
        for i in range(len(annotator_agreement_with_consensus)):
            labels = nontied_labels_multiannotator[:, i]
            labels_mask = ~np.isnan(labels)
            if np.sum(labels_mask) == 0:
                annotator_agreement_with_consensus[i] = np.nan
            else:
                annotator_agreement_with_consensus[i] = np.mean(
                    labels[labels_mask] == nontied_majority_vote_label[labels_mask]
                )

        # impute average annotator accuracy for any annotator that do not overlap with consensus
        nan_mask = np.isnan(annotator_agreement_with_consensus)
        avg_annotator_agreement = np.mean(annotator_agreement_with_consensus[~nan_mask])
        annotator_agreement_with_consensus[nan_mask] = avg_annotator_agreement

        for idx, label_mode in tied_idx.copy().items():
            label_quality_score = np.array(
                [
                    np.mean(
                        annotator_agreement_with_consensus[
                            np.where(labels_multiannotator[idx] == label)[0]
                        ]
                    )
                    for label in label_mode
                ]
            )
            max_score = np.where(label_quality_score == label_quality_score.max())[0]
            if len(max_score) == 1:
                majority_vote_label[idx] = label_mode[max_score[0]]
                del tied_idx[idx]
            else:
                tied_idx[idx] = label_mode[max_score]

    # if still tied, break by random selection
    if len(tied_idx) > 0:
        warnings.warn(
            f"breaking ties of examples {list(tied_idx.keys())} by random selection, you may want to set seed for reproducability"
        )
        for idx, label_mode in tied_idx.items():
            majority_vote_label[idx] = np.random.choice(label_mode)

    if verbose:
        # check if any classes no longer appear in the set of consensus labels
        check_consensus_label_classes(
            labels_multiannotator=labels_multiannotator,
            consensus_label=majority_vote_label,
            consensus_method="majority_vote",
        )

    return majority_vote_label.astype(int)

# --- from cleanlab__cleanlab::cleanlab/multiannotator.py::get_active_learning_scores_ensemble ---
def get_active_learning_scores_ensemble(
    labels_multiannotator: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    pred_probs: Optional[np.ndarray] = None,
    pred_probs_unlabeled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an ActiveLab quality score for each example in the dataset, based on predictions from an ensemble of models.

    This function is similar to `~cleanlab.multiannotator.get_active_learning_scores` but allows for an
    ensemble of multiple classifier models to be trained and will aggregate predictions from the models to compute the ActiveLab quality score.

    Parameters
    ----------
    labels_multiannotator : pd.DataFrame or np.ndarray
        Multiannotator labels in the same format expected by `~cleanlab.multiannotator.get_active_learning_scores`.
        This argument is optional if ``pred_probs`` is not provided (in cases where you only provide ``pred_probs_unlabeled`` to get active learning scores for unlabeled examples).
    pred_probs : np.ndarray
        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from the ensemble models.
        Note that this function also works with datasets where there is only one annotator (M=1).
        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
        This argument is optional if you only want to get active learning scores for unlabeled examples (pass in ``pred_probs_unlabeled`` instead).
    pred_probs_unlabeled : np.ndarray, optional
        An array of shape ``(P, N, K)`` where P is the number of models, consisting of predicted class probabilities from a trained classifier model
        for examples that have no annotated labels so far (but which we may want to label in the future, and hence compute active learning quality scores for).
        Each set of predicted probabilities with shape ``(N, K)`` is in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`.
        This argument is optional if you only want to get active learning scores for labeled examples (pass in ``pred_probs`` instead).

    Returns
    -------
    active_learning_scores : np.ndarray
        Similar to output as :py:func:`get_label_quality_scores <cleanlab.multiannotator.get_label_quality_scores>`.
    active_learning_scores_unlabeled : np.ndarray
        Similar to output as :py:func:`get_label_quality_scores <cleanlab.multiannotator.get_label_quality_scores>`.

    See Also
    --------
    get_active_learning_scores
    """

    assert_valid_pred_probs(
        pred_probs=pred_probs, pred_probs_unlabeled=pred_probs_unlabeled, ensemble=True
    )

    # compute multiannotator stats if labeled data is provided
    if pred_probs is not None:
        if labels_multiannotator is None:
            raise ValueError(
                "labels_multiannotator cannot be None when passing in pred_probs. ",
                "You can either provide labels_multiannotator to obtain active learning scores for the labeled examples, "
                "or just pass in pred_probs_unlabeled to get active learning scores for unlabeled examples.",
            )

        if isinstance(labels_multiannotator, pd.DataFrame):
            labels_multiannotator = (
                labels_multiannotator.replace({pd.NA: np.nan}).astype(float).to_numpy()
            )
        elif not isinstance(labels_multiannotator, np.ndarray):
            raise ValueError(
                "labels_multiannotator must be either a NumPy array or Pandas DataFrame."
            )

        # check that labels_multiannotator is a 2D array
        if labels_multiannotator.ndim != 2:
            raise ValueError(
                "labels_multiannotator must be a 2D array or dataframe, "
                "each row represents an example and each column represents an annotator."
            )

        num_classes = get_num_classes(pred_probs=pred_probs[0])

        # if all examples are only labeled by a single annotator
        if (np.sum(~np.isnan(labels_multiannotator), axis=1) == 1).all():
            # do not temp scale for single annotator case, temperature is defined here for later use
            optimal_temp = np.full(len(pred_probs), 1.0)

            assert_valid_inputs_multiannotator(
                labels_multiannotator, pred_probs, ensemble=True, allow_single_label=True
            )

            avg_pred_probs = np.mean(pred_probs, axis=0)
            consensus_label = get_majority_vote_label(
                labels_multiannotator=labels_multiannotator,
                pred_probs=avg_pred_probs,
                verbose=False,
            )
            quality_of_consensus_labeled = get_label_quality_scores(consensus_label, avg_pred_probs)
            model_weight = np.full(len(pred_probs), 1)
            annotator_weight = np.full(labels_multiannotator.shape[1], 1)
            avg_annotator_weight = np.mean(annotator_weight)

        # examples are annotated by multiple annotators
        else:
            optimal_temp = np.full(len(pred_probs), np.nan)
            for i, curr_pred_probs in enumerate(pred_probs):
                curr_optimal_temp = find_best_temp_scaler(labels_multiannotator, curr_pred_probs)
                pred_probs[i] = temp_scale_pred_probs(curr_pred_probs, curr_optimal_temp)
                optimal_temp[i] = curr_optimal_temp

            multiannotator_info = get_label_quality_multiannotator_ensemble(
                labels_multiannotator,
                pred_probs,
                return_annotator_stats=False,
                return_detailed_quality=False,
                return_weights=True,
            )

            quality_of_consensus_labeled = multiannotator_info["label_quality"][
                "consensus_quality_score"
            ]
            model_weight = multiannotator_info["model_weight"]
            annotator_weight = multiannotator_info["annotator_weight"]
            avg_annotator_weight = np.mean(annotator_weight)

        # compute scores for labeled data
        active_learning_scores = np.full(len(labels_multiannotator), np.nan)
        for i, annotator_labels in enumerate(labels_multiannotator):
            active_learning_scores[i] = np.average(
                (quality_of_consensus_labeled[i], 1 / num_classes),
                weights=(
                    np.sum(annotator_weight[~np.isnan(annotator_labels)]) + np.sum(model_weight),
                    avg_annotator_weight,
                ),
            )

    # no labeled data provided so do not estimate temperature and model/annotator weights
    elif pred_probs_unlabeled is not None:
        num_classes = get_num_classes(pred_probs=pred_probs_unlabeled[0])
        optimal_temp = np.full(len(pred_probs_unlabeled), 1.0)
        model_weight = np.full(len(pred_probs_unlabeled), 1)
        avg_annotator_weight = 1
        active_learning_scores = np.array([])

    else:
        raise ValueError(
            "pred_probs and pred_probs_unlabeled cannot both be None, specify at least one of the two."
        )

    # compute scores for unlabeled data
    if pred_probs_unlabeled is not None:
        for i in range(len(pred_probs_unlabeled)):
            pred_probs_unlabeled[i] = temp_scale_pred_probs(
                pred_probs_unlabeled[i], optimal_temp[i]
            )

        avg_pred_probs_unlabeled = np.mean(pred_probs_unlabeled, axis=0)
        consensus_label_unlabeled = get_majority_vote_label(
            np.argmax(pred_probs_unlabeled, axis=2).T,
            avg_pred_probs_unlabeled,
        )
        modified_pred_probs_unlabeled = np.average(
            np.concatenate(
                (
                    pred_probs_unlabeled,
                    np.full(pred_probs_unlabeled.shape[1:], 1 / num_classes)[np.newaxis, :, :],
                )
            ),
            weights=np.concatenate((model_weight, np.array([avg_annotator_weight]))),
            axis=0,
        )

        active_learning_scores_unlabeled = get_label_quality_scores(
            consensus_label_unlabeled, modified_pred_probs_unlabeled
        )
    else:
        active_learning_scores_unlabeled = np.array([])

    return active_learning_scores, active_learning_scores_unlabeled
