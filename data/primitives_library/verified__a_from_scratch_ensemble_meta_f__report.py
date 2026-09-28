from ensemblekit.validation import cross_val_score, np

__all__ = ["compare_models", "format_comparison"]


def compare_models(model_factories, X, y, cv=5) -> dict:
    result = {}

    for name, factory in model_factories.items():
        if not callable(factory):
            raise TypeError("each model factory must be callable")

        scores = np.asarray(cross_val_score(factory, X, y, cv=cv), dtype=float)
        result[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores,
        }

    return result


def format_comparison(result: dict) -> str:
    def sort_key(item):
        name, stats = item
        mean = float(stats.get("mean", float("nan")))
        if np.isnan(mean):
            return (1, 0.0, str(name))
        return (0, -mean, str(name))

    lines = []
    for name, stats in sorted(result.items(), key=sort_key):
        mean = float(stats["mean"])
        std = float(stats["std"])
        lines.append(f"{name}: {mean:.4f} +/- {std:.4f}")

    return "\n".join(lines)