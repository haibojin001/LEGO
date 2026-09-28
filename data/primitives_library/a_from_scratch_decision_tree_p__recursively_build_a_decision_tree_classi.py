def build_tree(df, features, target, method):
    labels = df[target].tolist()

    if labels.count(labels[0]) == len(labels):
        return labels[0]  # Pure leaf
    if not features:
        return Counter(labels).most_common(1)[0][0] 

    if method == "entropy":
        gains = {f: info_gain(df, f, target) for f in features}
        best_feature = max(gains, key=gains.get)
    else:
        ginis = {f: gini_index(df, f, target) for f in features}
        best_feature = min(ginis, key=ginis.get)

    tree = {best_feature: {}}

    for v in df[best_feature].unique():
        subset = df[df[best_feature] == v].drop(columns=[best_feature])
        sub_features = [f for f in features if f != best_feature]
        tree[best_feature][v] = build_tree(subset, sub_features, target, method)

    return tree