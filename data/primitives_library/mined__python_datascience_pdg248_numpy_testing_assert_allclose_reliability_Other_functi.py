# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg248::numpy.testing.assert_allclose+reliability.Other_functions.make_ALT_data+warnings.filterwarnings
# name: numpy_reliability_primitive
# summary: Uses numpy.testing.assert_allclose, reliability.Other_functions.make_ALT_data, warnings.filterwarnings across 4 repos
# anchor_symbols: ['numpy.testing.assert_allclose', 'reliability.Other_functions.make_ALT_data', 'warnings.filterwarnings']
# observed in 4 repos: ['MatthewReid854__reliability', 'apachecn__python_data_analysis_and_mining_action', 'cleanlab__cleanlab', 'zama-ai__concrete-ml']...

# --- from cleanlab__cleanlab::tests/test_object_detection.py::test_badloc_low_probability_threshold ---
def test_badloc_low_probability_threshold():
    prediction = predictions[3].copy()
    label = labels[3].copy()
    label["bboxes"] = np.append(label["bboxes"], [label["bboxes"][-1]], axis=0)
    label["labels"] = np.append(label["labels"], (label["labels"][-1] + 1) % 10)
    score = compute_badloc_box_scores(
        labels=[label], predictions=[prediction], low_probability_threshold=1.0
    )[0]
    assert np.allclose(score, np.ones_like(score), atol=1e-2)

# --- from apachecn__python_data_analysis_and_mining_action::chapter13/code.py::programmer_4 ---
def programmer_4():
    x0 = np.array([3152063, 2213050, 4050122,
                   5265142, 5556619, 4772843, 9463330])
    f, a, b, x00, C, P = GM11(x0)
    print(a, b, x00, C, P)
    print(u'2014年、2015年的预测结果分别为：\n%0.2f万元和%0.2f万元' % (f(8), f(9)))
    print(u'后验差比值为：%0.4f' % C)
    p = pd.DataFrame(x0, columns=["y"], index=range(2007, 2014))
    p.loc[2014] = None
    p.loc[2015] = None
    p["y_pred"] = [f(i) for i in range(1, 10)]
    p["y_pred"] = p["y_pred"].round(2)
    p.index = pd.to_datetime(p.index, format="%Y")

    p.plot(style=["b-o", "r-*"], xticks=p.index)
    plt.show()

# --- from cleanlab__cleanlab::tests/test_object_detection.py::test_swap_only_overlap_labels ---
def test_swap_only_overlap_labels(overlapping_label_check):
    prediction = predictions[3].copy()
    label = labels[3].copy()
    label["bboxes"] = np.append(label["bboxes"], [label["bboxes"][-1]], axis=0)
    label["labels"] = np.append(label["labels"], (label["labels"][-1] + 1) % 10)
    score = compute_swap_box_scores(
        labels=[label], predictions=[prediction], overlapping_label_check=overlapping_label_check
    )[0]
    if overlapping_label_check:
        assert np.allclose(score, np.array([0.88, 1.0, 0.95, 0.96, 1.0, 0.0, 0.0]), atol=1e-2)
    else:
        assert np.allclose(score, np.array([0.88, 1.0, 0.95, 0.96, 1.0, 0.88, 0.0]), atol=1e-2)

# --- from apachecn__python_data_analysis_and_mining_action::chapter13/code.py::GM11 ---
def GM11(x0):
    # 1-AGO序列, 累计求和
    x1 = np.cumsum(x0)
    # 紧邻均值（ＭＥＡＮ）生成序列
    z1 = (x1[:-1] + x1[1:]) / 2.0
    z1 = z1.reshape(len(z1), 1)
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Yn = x0[1:].reshape((len(x0) - 1, 1))
    # 矩阵计算，计算参数
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn)
    # 还原值

    f = lambda k: (x0[0] - b / a) * np.exp(-a * (k - 1)) - (x0[0] - b / a) * np.exp(-a * (k - 2))

    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))
    C = delta.std() / x0.std()
    P = 1.0 * (np.abs(delta - delta.mean()) <
               0.6745 * x0.std()).sum() / len(x0)
    # 灰度预测函数、a、b、首项、方差比、小残差概率

    return f, a, b, x0[0], C, P

# --- from MatthewReid854__reliability::tests/test_ALT_fitters.py::test_Fit_Exponential_Power ---
def test_Fit_Exponential_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Power', a=5e15, n=-4, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Power(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1970299637780768.8, rtol=rtol, atol=atol)
    assert_allclose(model.n, -3.831313136385626, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6314.7161417145035, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6322.083302623412, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3155.33786883705, rtol=rtol, atol=atol)

# --- from MatthewReid854__reliability::tests/test_ALT_fitters.py::test_Fit_Exponential_Eyring ---
def test_Fit_Exponential_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(distribution='Exponential', life_stress_model='Eyring', a=1500, c=-10, stress_1=[500, 400, 350], number_of_samples=100, fraction_censored=0.2, seed=1)
    model = Fit_Exponential_Eyring(failures=data.failures, failure_stress=data.failure_stresses, right_censored=data.right_censored, right_censored_stress=data.right_censored_stresses, use_level_stress=300, show_life_stress_plot=False, show_probability_plot=False, print_results=False)
    assert_allclose(model.a, 1428.4686331863793, rtol=rtol, atol=atol)
    assert_allclose(model.c, -10.259884009475353, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4200.055398999253, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4207.422559908162, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2098.0074974794247, rtol=rtol, atol=atol)

# --- from zama-ai__concrete-ml::src/concrete/ml/pandas/_operators.py::encrypted_left_right_join ---
def encrypted_left_right_join(
    left_encrypted,
    right_encrypted,
    server: Server,
    how: str,
    on: Optional[str],  # pylint: disable=invalid-name
) -> numpy.ndarray:
    """Compute a left/right join in FHE between two encrypted data-frames using Pandas parameters.

    Note that for now, only a left and right join is implemented. Additionally, only some Pandas
    parameters are supported, and joining on multiple columns is not available.

    The algorithm benefits from Concrete Python's composability feature. The idea is that for loops
    are done in the clear, meaning positional indexes are not encrypte and only the data is. In the
    case of a left merge, we need to select the encrypted value from the right data-frame for a
    given (left) row and (right) column position. In order to do that, a for loop goes through
    the right rows and runs the FHE circuit in a composable manner. The goal is to basically
    multiply the right column values with a mask which contains a single 1 at the row position where
    the left and right key matches, and then sum everything to retrieve the selected value. The
    main benefit of using composability instead of a dict mult and sum is that it does not require
    to know the number of columns and rows at compilation time. More details can be found in the
    '_development.py' file.

    Args:
        left_encrypted (EncryptedDataFrame): The left encrypted data-frame.
        right_encrypted (EncryptedDataFrame): The right encrypted data-frame.
        server (Server): The Concrete server to use for running the computations in FHE.
        how (str): Type of merge to be performed, one of {'left', 'right'}.
            * left: use only keys from left frame, similar to a SQL left outer join;
            preserve key order.
            * right: use only keys from right frame, similar to a SQL right outer join;
            preserve key order.
        on (Optional[str]): Column name to join on. These must be found in both DataFrames. If it is
            None then this defaults to the intersection of the columns in both DataFrames.

    Returns:
        numpy.ndarray: The values representing the joined encrypted data-frame.
    """
    allowed_how = ["left", "right"]
    assert how in allowed_how, f"Parameter 'how' must be in {allowed_how}. Got {how}."

    # In case of a right merge, swap the input data-frames
    if how == "right":
        left_encrypted, right_encrypted = right_encrypted, left_encrypted

    joined_rows = []

    # Retrieve the left and right column's position on which keys to merge
    left_key_column_position = left_encrypted.column_names_to_position[on]
    right_key_column_position = right_encrypted.column_names_to_position[on]

    # Retrieve the number of useful rows and columns
    n_rows_left = left_encrypted.encrypted_values.shape[0]
    n_columns_right = right_encrypted.encrypted_values.shape[1]
    n_rows_right = right_encrypted.encrypted_values.shape[0]

    # Loop over the left data frame's number of rows (which will become the joined data frame's
    # number of rows)
    for i_left in range(n_rows_left):

        # For left merge, all left values are exactly equal to the left data-frame
        array_joined_i_left = left_encrypted.encrypted_values[i_left, :]

        # In case of a right merge, remove the column containing the keys on which to merge. This
        # avoid unnecessary FHE computations as the output keys will exactly match the one contained
        # in the (initial) left data-frame. The reason why this is needed only for the right merge
        # is because, in Pandas, this selected column is always kept on the output data-frame's
        # left side. The column is manually inserted back at the end of this function
        if how == "right":
            array_joined_i_left = numpy.delete(
                array_joined_i_left, left_key_column_position, axis=0
            )

        left_row_to_join = array_joined_i_left.tolist()

        # Retrieve the left data frame's key to merge on
        left_key = left_encrypted.encrypted_values[i_left, left_key_column_position]

        right_row_to_join = []

        # Loop over the right data-frame's number of columns
        for j_right in range(n_columns_right):

            # Skip the right's index column
            if j_right == right_key_column_position:
                continue

            # Default value is NaN
            right_value_to_join = right_encrypted.encrypted_nan

            # Loop over the right data-frame's number of rows in order to check if one row's key
            # matches the on-going left key
            for i_right in range(n_rows_right):

                # Retrieve the right data-frame's value to sum if both keys match
                value_to_put_right = right_encrypted.encrypted_values[i_right, j_right]

                # Retrieve the right data frame's key to merge on
                right_key = right_encrypted.encrypted_values[i_right, right_key_column_position]

                merge_inputs = (right_value_to_join, value_to_put_right, left_key, right_key)

                # Run the FHE execution:
                # - on the first iteration, this is applied on a 0 (representing a NaN) and the
                #   right data-frame's value
                # - on the following iterations, this is applied between the previous accumulated
                # value and the right data-frame's value.
                # Basically, if both keys match, the function adds the accumulated value with the
                # right data-frame's value. If they don't, it just adds 0 to the accumulated value.
                # In practice, keys only match once throughout this very loop as keys are assumed to
                # be unique on both data-frames.
                right_value_to_join = server.run(
                    *merge_inputs, evaluation_keys=left_encrypted.evaluation_keys
                )

            right_row_to_join.append(right_value_to_join)

        # In case of a right merge, since data-frames wee initially swapped, swap back the values
        # when re-building the joined data-frame
        if how == "right":
            joined_row = right_row_to_join + left_row_to_join
        else:
            joined_row = left_row_to_join + right_row_to_join

        joined_rows.append(joined_row)

    array_joined = numpy.array(joined_rows)

    # In case of a right merge, as mentioned above, the column containing the right keys needs to be
    # manually re-inserted. This avoids unnecessary FHE computations
    if how == "right":
        array_joined = numpy.hstack(
            (
                array_joined[:, :right_key_column_position],
                left_encrypted.encrypted_values[
                    :, left_key_column_position : left_key_column_position + 1
                ],
                array_joined[:, right_key_column_position:],
            ),
        )

    return array_joined
