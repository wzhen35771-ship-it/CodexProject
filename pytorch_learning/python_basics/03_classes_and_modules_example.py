"""普通 Python 类、继承和模块示例；无自动求导或可训练参数注册。"""

import math
from math import sqrt


class FeatureDataset:
    def __init__(self, features, labels):
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        self.features = tuple(tuple(row) for row in features)
        self.labels = tuple(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"features": self.features[index], "label": self.labels[index]}


class ScaledFeatureDataset(FeatureDataset):
    def __init__(self, features, labels, scale):
        super().__init__(features, labels)
        self.scale = scale

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        scaled = tuple(value * self.scale for value in sample["features"])
        return {"features": scaled, "label": sample["label"]}


class ScalarLinear:
    def __init__(self, weight, bias=0.0):
        self.weight = weight
        self.bias = bias

    def forward(self, value):
        return self.weight * value + self.bias

    def __call__(self, value):
        # 只为讲解 Python 协议；不包含 nn.Module 的框架逻辑。
        return self.forward(value)


def main():
    features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    labels = [0, 1, 2]
    dataset = FeatureDataset(features, labels)
    scaled_dataset = ScaledFeatureDataset(features, labels, scale=2.0)
    print("dataset_length:", len(dataset))
    print("first_sample:", dataset[0])
    print("scaled_sample:", scaled_dataset[1])

    model = ScalarLinear(weight=2.0, bias=1.0)
    other = ScalarLinear(weight=4.0)
    print("model_attribute:", model.weight)
    print("forward:", model.forward(3.0))
    print("call:", model(3.0))
    model.weight = 8.0
    print("independent_weights:", model.weight, other.weight)

    print("module_import:", math.sqrt(9))
    print("name_import:", sqrt(16))

    assert len(dataset) == 3
    assert dataset[0] == {"features": (1.0, 2.0), "label": 0}
    assert scaled_dataset[1] == {"features": (6.0, 8.0), "label": 1}
    assert model(3.0) == model.forward(3.0) == 25.0
    assert other.weight == 4.0
    assert math.sqrt(9) == 3.0 and sqrt(16) == 4.0
    print("checks: passed")


if __name__ == "__main__":
    main()
