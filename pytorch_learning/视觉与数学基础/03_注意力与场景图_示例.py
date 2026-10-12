"""第三步的注意力练习：用小矩阵理解匹配、权重和信息融合。

全部在处理器上运行；不下载数据、图片或模型。
这些向量是教学数字，不代表真实物体的语义特征。
"""

import math

import torch


def compute_attention(queries, keys, values):
    """计算最基础的单头注意力，不含额外掩码和随机丢弃。"""
    # 查询形状：[查询数量, 特征维度]
    # 键形状：  [候选数量, 特征维度]
    # 值形状：  [候选数量, 值维度]
    raw_scores = queries @ keys.T
    feature_dim = queries.shape[-1]
    scaled_scores = raw_scores / math.sqrt(feature_dim)
    attention_weights = torch.softmax(scaled_scores, dim=-1)  # 每一行分别归一化。
    output = attention_weights @ values
    return raw_scores, scaled_scores, attention_weights, output


def show_attention_calculation(heading, queries, keys, values):
    print("\n" + heading)
    print("查询：", queries)
    print("键：", keys)
    print("值：", values)
    print("查询、键、值的形状：", list(queries.shape), list(keys.shape), list(values.shape))

    raw_scores, scaled_scores, attention_weights, output = compute_attention(queries, keys, values)
    print("第一步，查询与键做点积：", raw_scores)
    print("第二步，除以特征维度的平方根：", scaled_scores)
    print("第三步，转换成注意力权重：", attention_weights)
    print("各行权重之和：", attention_weights.sum(dim=-1))
    print("第四步，用权重融合值：", output)
    print("输出形状：", list(output.shape))
    return attention_weights, output


def main():
    torch.set_printoptions(precision=4, sci_mode=False)

    # float32 是浮点类型；device="cpu" 明确在处理器上计算。
    queries = torch.tensor([[1., 0.]], dtype=torch.float32, device="cpu")
    keys = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32, device="cpu")
    values = torch.tensor([[10., 0.], [0., 20.]], dtype=torch.float32, device="cpu")

    initial_weights, initial_output = show_attention_calculation("一、一个查询，两份信息", queries, keys, values)
    print("手算核对：权重约 [0.6698, 0.3302]；输出约 [6.6976, 6.6048]。")

    second_query = torch.tensor([[0., 1.]], dtype=torch.float32)
    show_attention_calculation("二、改变查询，匹配方向也会变化", second_query, keys, values)

    equal_score_query = torch.tensor([[1., 1.]], dtype=torch.float32)
    show_attention_calculation("三、两个匹配得分相同，权重各为一半", equal_score_query, keys, values)

    changed_values = torch.tensor([[10., 0.], [0., 100.]], dtype=torch.float32)
    updated_weights, updated_output = show_attention_calculation("四、只改变值，观察权重与输出", queries, keys, changed_values)
    print("权重是否保持相同：", "是" if torch.allclose(initial_weights, updated_weights) else "否")
    print("原来输出：", initial_output)
    print("改变值后的输出：", updated_output)
    print("原因：匹配使用查询和键；融合内容来自值。")

    query_batch = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32)
    show_attention_calculation("五、两个查询同时计算，每行得到自己的结果", query_batch, keys, values)

    print("\n六、自注意力的输入来源相同，学习投影可以不同")
    input_features = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32)
    # 以下投影是教学固定数字；真实模型中它们是通过训练学习的参数。
    query_projection = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32)
    key_projection = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float32)
    value_projection = torch.tensor([[10., 0.], [0., 20.]], dtype=torch.float32)
    projected_queries = input_features @ query_projection
    projected_keys = input_features @ key_projection
    projected_values = input_features @ value_projection
    show_attention_calculation("同一组输入经过三种投影", projected_queries, projected_keys, projected_values)
    print("来源相同，并不保证查询、键、值的数值相同。")

    print("\n理解检查")
    print("1. 如果查询改成 [0, 0]，两个权重和输出分别是多少？")
    print("2. 为什么原始得分为 0 的候选仍有非零权重？")
    print("3. 值有三个分量时，输出会有几个分量？")
    print("4. 注意力输出是融合特征，还需要分类层才能得到关系预测。")
    print("5. 本文件没有训练模型；也没有实现 STTran 或 ImparTail。")


if __name__ == "__main__":
    main()
