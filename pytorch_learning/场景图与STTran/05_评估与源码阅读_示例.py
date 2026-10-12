"""第五天：人工候选的排序、构图、框匹配与召回率。

仅演示评估核心思想，采用连续坐标 IoU，不替代项目官方评估器。
"""

from collections import defaultdict


def calculate_iou(box_a, box_b):
    intersection_width = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    intersection_height = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection_area = intersection_width * intersection_height
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def create_record(triplet, subject_box, object_box, relation_score=1.0, group="", object_pair=(0, 1),
             subject_score=1.0, object_score=1.0):
    return {
        "三元组": triplet,
        "主体框": subject_box,
        "客体框": object_box,
        "关系分数": relation_score,
        "主体分数": subject_score,
        "客体分数": object_score,
        "组": group,
        "物体对": object_pair,
    }


def compute_combined_score(candidate):
    return candidate["主体分数"] * candidate["关系分数"] * candidate["客体分数"]


def matches_ground_truth(candidate, ground_truth):
    return (
        candidate["三元组"] == ground_truth["三元组"]
        and calculate_iou(candidate["主体框"], ground_truth["主体框"]) >= 0.5
        and calculate_iou(candidate["客体框"], ground_truth["客体框"]) >= 0.5
    )


def recall_at_k(candidates, ground_truths, k):
    matched_indices = set()
    sorted_candidates = sorted(candidates, key=compute_combined_score, reverse=True)
    for candidate in sorted_candidates[:k]:
        for ground_truth_index, ground_truth in enumerate(ground_truths):
            if matches_ground_truth(candidate, ground_truth):
                matched_indices.add(ground_truth_index)
    if not ground_truths:
        raise ValueError("这个教学例子要求每帧至少有一条真值。")
    return len(matched_indices) / len(ground_truths), matched_indices


def filter_candidates(candidates, strategy, threshold=0.65):
    if strategy == "无约束":
        return list(candidates)
    candidates_by_pair_and_group = defaultdict(list)
    for candidate in candidates:
        candidates_by_pair_and_group[(candidate["物体对"], candidate["组"])].append(candidate)
    retained_candidates = []
    for (_, group), group_candidates in candidates_by_pair_and_group.items():
        if strategy == "有约束" or group == "注意":
            retained_candidates.append(max(group_candidates, key=lambda candidate_item: candidate_item["关系分数"]))
        elif strategy == "半约束":
            retained_candidates.extend(candidate_item for candidate_item in group_candidates if candidate_item["关系分数"] > threshold)
        else:
            raise ValueError("未知构图策略")
    return retained_candidates


def main():
    person_box = [0, 0, 20, 60]
    cup_box = [30, 20, 40, 40]
    incorrect_cup_box = [70, 20, 80, 40]
    frame_0_ground_truths = [
        create_record(("人", "看着", "杯子"), person_box, cup_box),
        create_record(("人", "拿着", "杯子"), person_box, cup_box),
        create_record(("杯子", "在前方", "人"), cup_box, person_box),
    ]
    frame_0_candidates = [
        create_record(("人", "不看着", "杯子"), person_box, cup_box, 0.9, "注意"),
        create_record(("人", "拿着", "杯子"), person_box, cup_box, 0.8, "接触"),
        create_record(("杯子", "在前方", "人"), cup_box, person_box, 0.7, "空间"),
        create_record(("人", "看着", "杯子"), person_box, cup_box, 0.6, "注意"),
    ]
    frame_1_ground_truths = [create_record(("人", "拿着", "杯子"), person_box, cup_box)]
    frame_1_candidates = [
        create_record(("人", "拿着", "杯子"), person_box, incorrect_cup_box, 0.95, "接触", (0, 1)),
        create_record(("人", "拿着", "杯子"), person_box, cup_box, 0.75, "接触", (0, 2)),
    ]

    print("一、综合得分：0.9×0.8×0.7 =", round(0.9 * 0.8 * 0.7, 6))
    print("\n二、第0帧：按分数排序，再看前K个预测")
    for k in [1, 2, 3, 4]:
        recall_score, matched_indices = recall_at_k(frame_0_candidates, frame_0_ground_truths, k)
        print(f"R@{k}={recall_score:.4f}；找回的真值编号：{sorted(matched_indices)}")

    print("\n三、文字正确而框错误的候选")
    print("杯子框交并比：", calculate_iou(incorrect_cup_box, cup_box))
    print("是否匹配：", matches_ground_truth(frame_1_candidates[0], frame_1_ground_truths[0]))
    frame_0_recall, frame_0_matched_indices = recall_at_k(frame_0_candidates, frame_0_ground_truths, 2)
    frame_1_recall, frame_1_matched_indices = recall_at_k(frame_1_candidates, frame_1_ground_truths, 2)
    mean_frame_recall = (frame_0_recall + frame_1_recall) / 2
    pooled_instance_recall = (len(frame_0_matched_indices) + len(frame_1_matched_indices)) / (len(frame_0_ground_truths) + len(frame_1_ground_truths))
    print(f"第0帧 R@2={frame_0_recall:.4f}；第1帧 R@2={frame_1_recall:.4f}")
    print(f"帧平均={mean_frame_recall:.4f}；实例汇总={pooled_instance_recall:.4f}")

    print("\n四、构图策略改变保留候选")
    for strategy in ["无约束", "有约束", "半约束"]:
        retained_candidates = filter_candidates(frame_0_candidates, strategy)
        print(strategy, "保留：", [candidate_item["三元组"] for candidate_item in retained_candidates])
        recall_score, _ = recall_at_k(retained_candidates, frame_0_ground_truths, 4)
        print(f"使用全部保留候选时 R@4={recall_score:.4f}")

    print("\n五、重复预测不会把同一条真值计算两次")
    duplicate_candidates = [frame_0_candidates[1], frame_0_candidates[1]]
    recall_score, matched_indices = recall_at_k(duplicate_candidates, frame_0_ground_truths, 2)
    print(f"两个重复‘拿着’预测：R@2={recall_score:.4f}；不同真值数={len(matched_indices)}")
    print("\n练习：将第0帧‘看着’分数改为0.95，先预测排序与有约束结果。")


if __name__ == "__main__":
    main()

