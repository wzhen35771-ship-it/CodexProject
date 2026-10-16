"""纯 Python 数据流示例：观察批次与预测，不计算梯度或更新参数。"""

import argparse


class RelationDataset:
    """练习可供 Dataset 使用的 __len__ / __getitem__ 约定；不是 torch 子类。"""

    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample["features"], sample["label"]


def make_samples():
    # dx/dy = 主语中心 - 宾语中心；图像 y 轴向下增大。
    return [
        {"features": (-0.8, 0.0), "label": 0},
        {"features": (-0.6, 0.1), "label": 0},
        {"features": (0.8, 0.0), "label": 1},
        {"features": (0.6, 0.1), "label": 1},
        {"features": (0.0, -0.8), "label": 2},
        {"features": (0.1, -0.6), "label": 2},
    ]


def make_batches(dataset, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    batches = []
    for start in range(0, len(dataset), batch_size):
        batch_features, batch_labels = [], []
        for index in range(start, min(start + batch_size, len(dataset))):
            features, label = dataset[index]
            batch_features.append(features)
            batch_labels.append(label)
        batches.append((batch_features, batch_labels))
    return batches


def predict_by_rule(features):
    dx, dy = features
    if dy < 0 and abs(dy) > abs(dx):
        return 2
    return 0 if dx < 0 else 1


def accuracy(predictions, labels):
    if len(predictions) != len(labels) or not labels:
        raise ValueError("预测和标签必须等长且非空")
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    return correct / len(labels)


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description="用纯 Python 追踪训练脚本需要的数据与控制流")
    parser.add_argument("--epochs", type=positive_integer, default=2)
    parser.add_argument("--batch-size", type=positive_integer, default=4)
    arguments = parser.parse_args()
    config = {"epochs": arguments.epochs, "batch_size": arguments.batch_size}

    dataset = RelationDataset(make_samples())
    batches = make_batches(dataset, config["batch_size"])
    expected_batch_count = (len(dataset) + config["batch_size"] - 1) // config["batch_size"]
    if len(batches) != expected_batch_count:
        raise RuntimeError("批次划分数量不正确")
    print("配置：", config)
    print("样本数：", len(dataset), "每轮批次数：", len(batches))
    print("首个样本 features / label：", dataset[0])

    processed_batches = 0
    for epoch in range(1, config["epochs"] + 1):
        all_predictions, all_labels = [], []
        for batch_index, (batch_features, batch_labels) in enumerate(batches, start=1):
            predictions = [predict_by_rule(features) for features in batch_features]
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels)
            processed_batches += 1
            print(f"epoch={epoch} batch={batch_index} 样本数={len(batch_labels)} 预测={predictions}")
        print(f"本轮规则预测准确率：{accuracy(all_predictions, all_labels):.1%}")

    if processed_batches != config["epochs"] * len(batches):
        raise RuntimeError("循环执行次数不正确")
    print("共处理批次数：", processed_batches)
    print("参数更新次数：0；固定规则没有训练，也没有损失或梯度。")
    print("换成真实训练时，更新次数要在 optimizer.step() 实际执行后计数。")


if __name__ == "__main__":
    main()
