"""综合作业：实现三个 TODO 后，运行本文件 --check。

纯 Python；不需要安装 PyTorch。未完成时返回退出码 2，并说明原因。
"""

import argparse


def count_batches(sample_count, batch_size):
    """不丢弃尾批：sample_count >= 0，batch_size >= 1；不满足时抛 ValueError。"""
    # TODO: 完成参数检查和批次数计算。
    if sample_count>=0 and batch_size>=1:
        return (sample_count + batch_size - 1) // batch_size
    raise ValueError("cuowu")


def make_batches(samples, batch_size):
    """按顺序把列表切成批次，保留尾批，返回列表；batch_size < 1 时抛 ValueError。"""
    # TODO: 不修改原 samples；空列表返回 []。
    sample=list(samples)
    if batch_size < 1:
        raise ValueError("batch_size < 1")
    return [sample[i:i + batch_size] for i in range(0, len(sample), batch_size)]


def compute_accuracy(predictions, labels):
    """等长且非空时返回准确率；长度不等或为空时抛 ValueError。"""
    # TODO: 准确率应是 0 到 1 的数，不是百分比字符串。
    if len(predictions) != len(labels) or len(predictions) == 0:
        raise ValueError("长度不等或为空")
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)

def expect_value_error(function, *arguments):
    try:
        function(*arguments)
    except ValueError:
        return
    raise AssertionError(f"{function.__name__}{arguments} 应抛出 ValueError")


def run_checks():
    assert count_batches(12, 4) == 3
    assert count_batches(7, 3) == 3
    assert count_batches(2, 8) == 1
    assert count_batches(0, 4) == 0
    expect_value_error(count_batches, -1, 4)
    expect_value_error(count_batches, 4, 0)

    original = ["a", "b", "c", "d", "e"]
    assert make_batches(original, 2) == [["a", "b"], ["c", "d"], ["e"]]
    assert original == ["a", "b", "c", "d", "e"]
    assert make_batches([], 2) == []
    assert make_batches([1, 2], 8) == [[1, 2]]
    expect_value_error(make_batches, original, 0)

    assert abs(compute_accuracy([0, 1, 1], [0, 1, 2]) - 2 / 3) < 1e-12
    assert compute_accuracy([0, 0], [1, 1]) == 0.0
    assert compute_accuracy([2, 1], [2, 1]) == 1.0
    expect_value_error(compute_accuracy, [], [])
    expect_value_error(compute_accuracy, [0], [0, 1])
    print("全部自检通过。接着用自己的数据追踪批次，并解释实际参数更新的位置。")


def main():
    parser = argparse.ArgumentParser(description="Python 综合练习；完成 TODO 后使用 --check")
    parser.add_argument("--check", action="store_true", help="检查三个 TODO 的正常情况与边界情况")
    arguments = parser.parse_args()
    if not arguments.check:
        print("请完成 count_batches、make_batches、compute_accuracy，再运行本文件 --check。")
        return 0
    try:
        run_checks()
    except NotImplementedError as error:
        print(f"作业尚未完成：{error}。参考配套讲义的作业要求，不要把此文件当作已完成示例。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
