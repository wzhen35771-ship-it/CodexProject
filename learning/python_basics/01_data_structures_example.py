"""普通 Python 数据组织练习；无 PyTorch、无文件写入。"""

from copy import deepcopy


def main():
    learning_rate = 0.01
    batch_size = 4
    is_training = True
    model_name = "baseline"
    checkpoint = None
    values = [learning_rate, batch_size, is_training, model_name, checkpoint]
    print("types:", [type(value).__name__ for value in values])

    features = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    labels = [0, 1, 2]
    sample = (features[0], labels[0])
    config = {"batch_size": 2, "learning_rate": learning_rate}
    print("sample:", sample)
    print("config:", config["batch_size"], config.get("seed", 0))
    print("index/slice:", labels[-1], labels[1:3], labels[::2])
    print("feature_rows:", len(features))

    original = [0, 1]
    alias = original
    alias.append(2)
    copied = original.copy()
    copied[0] = 9
    print("same_object:", alias is original)
    print("flat_copy:", original, copied)

    rows = [[1, 2], [3, 4]]
    shallow = rows.copy()
    independent = deepcopy(rows)
    shallow[0][0] = 99
    independent[1][0] = 88
    print("nested_original:", rows)
    print("nested_shallow:", shallow)
    print("nested_deep:", independent)

    assert [type(value).__name__ for value in values] == [
        "float", "int", "bool", "str", "NoneType"
    ]
    assert len(features) == len(labels) == 3
    assert sample == ([0.1, 0.2], 0)
    assert original == [0, 1, 2] and copied == [9, 1, 2]
    assert shallow is not rows and shallow[0] is rows[0]
    assert rows == [[99, 2], [3, 4]]
    assert independent == [[1, 2], [88, 4]]
    print("checks: passed")


if __name__ == "__main__":
    main()
