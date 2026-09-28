# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg562::matplotlib.pyplot.close+matplotlib.pyplot.subplots+seaborn.set
# name: matplotlib_seaborn_primitive
# summary: Uses matplotlib.pyplot.close, matplotlib.pyplot.subplots, seaborn.set across 4 repos
# anchor_symbols: ['matplotlib.pyplot.close', 'matplotlib.pyplot.subplots', 'seaborn.set']
# observed in 4 repos: ['ClimbsRocks__auto_ml', 'devAmoghS__Machine-Learning-with-Python', 'sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/report/report_deco.py::plot_data_hist ---
def plot_data_hist(data, title="title", bins=100, path=None):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))
    sns.distplot(data, bins=bins, color="m", ax=axs)
    axs.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close()

# --- from sberbank-ai-lab__LightAutoML::lightautoml/report/report_deco.py::plot_data_hist ---
def plot_data_hist(data, title="title", bins=100, path=None):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))
    sns.distplot(data, bins=bins, color="m", ax=axs)
    axs.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close()

# --- from sb-ai-lab__LightAutoML::lightautoml/report/report_deco.py::plot_feature_importance ---
def plot_feature_importance(feat_imp, path, features_max=100):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, features_max / 2.5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp[:features_max], ax=axs, color="m")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# --- from sberbank-ai-lab__LightAutoML::lightautoml/report/report_deco.py::plot_feature_importance ---
def plot_feature_importance(feat_imp, path, features_max=100):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, features_max / 2.5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp[:features_max], ax=axs, color="m")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# --- from devAmoghS__Machine-Learning-with-Python::prec_rec_curve.py::precision_recall_curve ---
def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        f1_score = (2 * precision * recall) / (precision + recall)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return precisions, recalls, f1_scores

# --- from ClimbsRocks__auto_ml::auto_ml/utils_scoring.py::advanced_scoring_regressors ---
def advanced_scoring_regressors(predictions, actuals, verbose=2, name=None):
    # pandas Series don't play nice here. Make sure our actuals list is indeed a list
    actuals = list(actuals)
    predictions = list(predictions)

    print('\n\n***********************************************')
    if name != None:
        print(name)
    print('Advanced scoring metrics for the trained regression model on this particular dataset:\n')

    # 1. overall RMSE
    print('Here is the overall RMSE for these predictions:')
    rmse = mean_squared_error(actuals, predictions)**0.5
    print(rmse)

    # 2. overall avg predictions
    print('\nHere is the average of the predictions:')
    print(sum(predictions) * 1.0 / len(predictions))

    # 3. overall avg actuals
    print('\nHere is the average actual value on this validation set:')
    print(sum(actuals) * 1.0 / len(actuals))

    # 2(a). median predictions
    print('\nHere is the median prediction:')
    print(np.median(predictions))

    # 3(a). median actuals
    print('\nHere is the median actual value:')
    print(np.median(actuals))

    # 4. avg differences (not RMSE)
    print('\nHere is the mean absolute error:')
    print(mean_absolute_error(actuals, predictions))

    print('\nHere is the median absolute error (robust to outliers):')
    print(median_absolute_error(actuals, predictions))

    print('\nHere is the explained variance:')
    print(explained_variance_score(actuals, predictions))

    print('\nHere is the R-squared value:')
    print(r2_score(actuals, predictions))

    # 5. pos and neg differences
    calculate_and_print_differences(predictions=predictions, actuals=actuals, name=name)

    actuals_preds = list(zip(actuals, predictions))
    # Sort by PREDICTED value, since this is what what we will know at the time we make a prediction
    actuals_preds.sort(key=lambda pair: pair[1])
    actuals_sorted = [act for act, pred in actuals_preds]
    predictions_sorted = [pred for act, pred in actuals_preds]

    if verbose > 2:
        print('Here\'s how the trained predictor did on each successive decile (ten percent chunk) of the predictions:')
        for i in range(1,11):
            print('\n**************')
            print('Bucket number:')
            print(i)
            # There's probably some fenceposting error here
            min_idx = int((i - 1) / 10.0 * len(actuals_sorted))
            max_idx = int(i / 10.0 * len(actuals_sorted))
            actuals_for_this_decile = actuals_sorted[min_idx:max_idx]
            predictions_for_this_decile = predictions_sorted[min_idx:max_idx]

            print('Avg predicted val in this bucket')
            print(sum(predictions_for_this_decile) * 1.0 / len(predictions_for_this_decile))
            print('Avg actual val in this bucket')
            print(sum(actuals_for_this_decile) * 1.0 / len(actuals_for_this_decile))
            print('RMSE for this bucket')
            print(mean_squared_error(actuals_for_this_decile, predictions_for_this_decile)**0.5)
            calculate_and_print_differences(predictions_for_this_decile, actuals_for_this_decile)

    print('')
    print('\n***********************************************\n\n')
    return rmse
