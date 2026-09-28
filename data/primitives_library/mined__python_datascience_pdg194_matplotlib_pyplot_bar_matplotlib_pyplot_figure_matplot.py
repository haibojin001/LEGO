# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg194::matplotlib.pyplot.bar+matplotlib.pyplot.figure+matplotlib.pyplot.savefig
# name: matplotlib_primitive
# summary: Uses matplotlib.pyplot.bar, matplotlib.pyplot.figure, matplotlib.pyplot.savefig, matplotlib.pyplot.tight_layout across 2 repos
# anchor_symbols: ['matplotlib.pyplot.bar', 'matplotlib.pyplot.figure', 'matplotlib.pyplot.savefig', 'matplotlib.pyplot.tight_layout', 'matplotlib.pyplot.xticks', 'matplotlib.pyplot.ylabel']
# observed in 2 repos: ['microsoft__RD-Agent', 'mljar__mljar-supervised']...

# --- from microsoft__RD-Agent::rdagent/app/benchmark/factor/analysis.py::Plotter.plot_data ---
def plot_data(data, file_name, title):
        plt.figure(figsize=(10, 10))
        plt.ylabel("Value")
        colors = ["#3274A1", "#E1812C", "#3A923A", "#C03D3E"]
        plt.bar(data["a"], data["b"], color=colors, capsize=5)
        for idx, row in data.iterrows():
            plt.text(idx, row["b"] + 0.01, f"{row['b']:.2f}", ha="center", va="bottom")
        plt.suptitle(title, y=0.98)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(file_name)

# --- from mljar__mljar-supervised::supervised/utils/learning_curves.py::LearningCurves.plot_single_iter ---
def plot_single_iter(learner_names, metric_name, model_path, colors):
        plt.figure(figsize=(10, 7))
        for ln in learner_names:
            df = pd.read_csv(
                os.path.join(model_path, f"{ln}_training.log"),
                names=["iteration", "train", "test"],
            )

            fold, repeat = learner_name_to_fold_repeat(ln)
            repeat_str = f" Reapeat {repeat+1}," if repeat is not None else ""
            plt.bar(
                f"Fold {fold+1},{repeat_str} train",
                df.train[0],
                color="white",
                edgecolor=colors[fold],
            )
            plt.bar(f"Fold {fold+1},{repeat_str} test", df.test[0], color=colors[fold])

        plt.ylabel(metric_name)
        plt.xticks(rotation=90)
        plt.tight_layout(pad=2.0)
        plot_path = os.path.join(model_path, LearningCurves.output_file_name)
        plt.savefig(plot_path)
        plt.close("all")
