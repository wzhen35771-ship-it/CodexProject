"""用普通数字演示样本配对、batch 循环和函数；不是训练模型。"""


def mean(values):
    if len(values) == 0:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def summarize_batch(features, labels, scale=1.0):
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    scores = [sum(feature) * scale for feature in features]
    return {"count": len(labels), "mean_score": mean(scores)}


def record(value, history=None):
    if history is None:
        history = []
    history.append(value)
    return history


def main():
    features = [[1, 2], [3, 4], [5, 6]]
    labels = [0, 1, 2]
    print("label_indices:", list(enumerate(labels)))
    print("pairs:", list(zip(features, labels)))
    print("positive_labels:", [label for label in labels if label > 0])
    print("zip_truncates:", list(zip([0, 1, 2], ["left", "right"])))

    summary = summarize_batch(features[:2], labels[:2], scale=0.5)
    print("scaled_summary:", summary)
    assert summary == {"count": 2, "mean_score": 2.5}

    independent_first = record(0.4)
    independent_second = record(0.2)
    print("independent_history:", independent_first, independent_second)
    assert independent_first == [0.4] and independent_second == [0.2]

    batch_size = 2
    history = []
    batch_counts = []
    for epoch in range(2):
        for start in range(0, len(features), batch_size):
            batch_features = features[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]
            summary = summarize_batch(batch_features, batch_labels)
            size_name = "full" if summary["count"] == batch_size else "partial"
            record(summary["mean_score"], history)
            batch_counts.append(summary["count"])
            print(
                f"epoch={epoch} start={start} count={summary['count']} "
                f"mean_score={summary['mean_score']:.1f} size={size_name}"
            )

    print("history:", history)
    assert batch_counts == [2, 1, 2, 1]
    assert history == [5.0, 11.0, 5.0, 11.0]
    assert len(list(zip([0, 1, 2], ["left", "right"]))) == 2
    print("checks: passed")


if __name__ == "__main__":
    main()
