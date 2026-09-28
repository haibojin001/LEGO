# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg144::matplotlib.pyplot.close+matplotlib.pyplot.figure+matplotlib.pyplot.legend
# name: matplotlib_pandas_primitive
# summary: Uses matplotlib.pyplot.close, matplotlib.pyplot.figure, matplotlib.pyplot.legend, matplotlib.pyplot.plot across 3 repos
# anchor_symbols: ['matplotlib.pyplot.close', 'matplotlib.pyplot.figure', 'matplotlib.pyplot.legend', 'matplotlib.pyplot.plot', 'matplotlib.pyplot.savefig', 'matplotlib.pyplot.tight_layout', 'matplotlib.pyplot.title', 'matplotlib.pyplot.xlabel', 'matplotlib.pyplot.ylabel', 'matplotlib.pyplot.ylim', 'pandas.read_csv']
# observed in 3 repos: ['microsoft__nni', 'mljar__mljar-supervised', 'refuel-ai__autolabel']...

# --- from mljar__mljar-supervised::supervised/utils/learning_curves.py::LearningCurves.plot_for_ensemble ---
def plot_for_ensemble(scores, metric_name, model_path):
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, len(scores) + 1), scores, label=f"Ensemble")
        plt.xlabel("#Iteration")
        plt.ylabel(metric_name)
        plt.legend(loc="best")
        plot_path = os.path.join(model_path, LearningCurves.output_file_name)
        plt.savefig(plot_path)
        plt.close("all")

# --- from refuel-ai__autolabel::src/autolabel/confidence.py::ConfidenceCalculator.compute_auroc ---
def compute_auroc(
        cls, match: List[int], confidence: List[float], plot: bool = False,
    ) -> Tuple[float, List[float]]:
        try:
            import sklearn
        except ImportError:
            raise ValueError(
                "Could not import sklearn python package. "
                "Please it install it with `pip install scikit-learn`.",
            )

        if len(set(match)) == 1:
            # ROC AUC score is not defined for a label list with
            # just one prediction
            return 1.0, [0]
        area = sklearn.metrics.roc_auc_score(match, confidence)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(match, confidence, pos_label=1)
        fpr, tpr, thresholds = (
            fpr[1:],
            tpr[1:],
            thresholds[1:],
        )  # first element is always support = 0. Can safely ignore.
        if plot:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ValueError(
                    "Could not import matplotlib python package. "
                    "Please it install it with `pip install matplotlib`.",
                )
            print(f"FPR: {fpr}")
            print(f"TPR: {tpr}")
            print(f"Thresholds: {thresholds}")
            print(
                f"Completion Rate: {[ConfidenceCalculator.compute_completion(confidence, i) for i in thresholds]}",
            )
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                label="ROC curve (area = %0.2f)" % area,
            )
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic example")
            plt.legend(loc="lower right")
            plt.savefig("AUROC_CURVE.png")
            plt.close()
        return area, thresholds

# --- from mljar__mljar-supervised::supervised/utils/learning_curves.py::LearningCurves.plot_iterations ---
def plot_iterations(
        learner_names, metric_name, model_path, colors, trees_in_iteration=None
    ):
        plt.figure(figsize=(10, 7))
        for ln in learner_names:
            df = pd.read_csv(
                os.path.join(model_path, f"{ln}_training.log"),
                names=["iteration", "train", "test"],
            )

            fold, repeat = learner_name_to_fold_repeat(ln)
            repeat_str = f" Reapeat {repeat+1}," if repeat is not None else ""
            # if trees_in_iteration is not None:
            #    df.iteration = df.iteration * trees_in_iteration
            any_none = np.sum(pd.isnull(df.train))
            if any_none == 0:
                plt.plot(
                    df.iteration,
                    df.train,
                    "--",
                    color=colors[fold],
                    label=f"Fold {fold+1},{repeat_str} train",
                )
            any_none = np.sum(pd.isnull(df.test))
            if any_none == 0:
                plt.plot(
                    df.iteration,
                    df.test,
                    color=colors[fold],
                    label=f"Fold {fold+1},{repeat_str} test",
                )

            
            if not df.test.isnull().values.any():
                best_iter = None
                if Metric.optimize_negative(metric_name):
                    best_iter = df.test.argmax()
                else:
                    best_iter = df.test.argmin()

                if best_iter is not None and best_iter != -1:
                    plt.axvline(best_iter, color=colors[fold], alpha=0.3)

        if trees_in_iteration is not None:
            plt.xlabel("#Trees")
        else:
            plt.xlabel("#Iteration")
        plt.ylabel(metric_name)

        # limit number of learners in the legend
        # too many will raise warnings
        if len(learner_names) <= 15:
            plt.legend(loc="best")

        plt.tight_layout(pad=2.0)
        plot_path = os.path.join(model_path, LearningCurves.output_file_name)
        plt.savefig(plot_path)
        plt.close("all")

# --- from microsoft__nni::examples/trials/benchmarking/automlbenchmark/parse_result_csv.py::generate_graphs ---
def generate_graphs(result_file_name):
    """
    Generate graphs describing performance statistics.
    The input result_file_name should be the path of the "results.csv" generated by automlbenchmark.
    For each task, this function outputs two graphs in the "reports/" directory located in the same 
    parent directory as "results.csv".
    The graph named task_foldx_1.jpg summarizes the best score each tuner gets after n trials. 
    The graph named task_foldx_2.jpg summarizes the score each tuner gets in each trial. 
    """
    markers = list(Line2D.markers.keys())
    result = pd.read_csv(result_file_name)
    scorelog_dir = result_file_name.replace('results.csv', 'scorelogs/')
    output_dir = result_file_name.replace('results.csv', 'reports/') 
    task_ids = result['id'].unique()
    for task_id in task_ids:
        task_results = result[result['id'] == task_id]
        task_name = task_results.task.unique()[0]
        folds = task_results['fold'].unique()        

        for fold in folds:            
            # load scorelog files
            trial_scores, best_scores = [], []
            tuners = list(task_results[task_results.fold == fold]['framework'].unique())
            for tuner in tuners:
                scorelog_name = '{}_{}_{}.csv'.format(tuner.lower(), task_name, fold)
                intermediate_scores = pd.read_csv(scorelog_dir + scorelog_name)
                bs = list(intermediate_scores['best_score'])
                ts = [(i+1, x) for i, x in enumerate(list(intermediate_scores['trial_score'])) if x != 0]
                best_scores.append([tuner, bs])
                trial_scores.append([tuner, ts])

            # generate the best score graph
            plt.figure(figsize=(16, 8))
            for i, (tuner, score) in enumerate(best_scores):
                plt.plot(score, label=tuner, marker=markers[i])
            plt.title('{} Fold {}'.format(task_name, fold))
            plt.xlabel("Number of Trials")
            plt.ylabel("Best Score")        
            plt.legend()
            plt.savefig(output_dir + '{}_fold{}_1.jpg'.format(task_name, fold))
            plt.close()

            # generate the trial score graph
            plt.figure(figsize=(16, 8))
            for i, (tuner, score) in enumerate(trial_scores):
                x = [l[0] for l in score]
                y = [l[1] for l in score] 
                plt.plot(x, y, label=tuner)      #, marker=markers[i])
            plt.title('{} Fold {}'.format(task_name, fold))
            plt.xlabel("Trial Number")
            plt.ylabel("Trial Score")        
            plt.legend()
            plt.savefig(output_dir + '{}_fold{}_2.jpg'.format(task_name, fold))
            plt.close()
