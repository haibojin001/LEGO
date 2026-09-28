def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    label_to_index = {l:i for i,l in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t,p in zip(y_true, y_pred):
        cm[label_to_index[t]][label_to_index[p]] += 1

    return cm, labels