"""视觉基础练习：图片张量、维度、预处理、物体框和特征。

所有数字都是手工构造的教学数据，没有调用真实物体检测模型。
只依赖已安装的 torch，不下载图片或权重，在 CPU 上即可运行。
"""

import torch


def single_box_iou(box_a, box_b):
    """计算两个正面积 xyxy 框的交并比，返回单元素张量。

    本函数采用连续边界坐标，宽高不加 1。
    """
    if box_a.shape != (4,) or box_b.shape != (4,):
        raise ValueError("每个框必须是一维张量，并且恰好有四个数字。")
    # 切片的前两个元素是左上边界，后两个是右下边界。
    size_a = box_a[2:] - box_a[:2]
    size_b = box_b[2:] - box_b[:2]
    if not bool((size_a > 0).all()) or not bool((size_b > 0).all()):
        raise ValueError("物体框必须有正的宽度和高度。")

    # 重合区域取更靠右、更靠下的左上边界。
    intersection_left_top = torch.maximum(box_a[:2], box_b[:2])
    # 重合区域取更靠左、更靠上的右下边界。
    intersection_right_bottom = torch.minimum(box_a[2:], box_b[2:])
    # 分离时边界差值可能为负，将其改成 0，表示没有面积交集。
    intersection_size = (intersection_right_bottom - intersection_left_top).clamp(min=0)
    intersection_area = intersection_size.prod()  # 交集宽乘以交集高。

    area_a = size_a.prod()
    area_b = size_b.prod()
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area


def xyxy_to_cxcywh(box):
    """将合法的 xyxy 框转换成中心横坐标、中心纵坐标、宽、高。"""
    center = (box[:2] + box[2:]) / 2
    size = box[2:] - box[:2]
    return torch.cat([center, size])


def main():
    torch.set_printoptions(precision=3, sci_mode=False)

    print("一、创建两行两列的 RGB 图片")
    print("颜色排列：第一行红、绿；第二行蓝、白。")
    image_hwc = torch.tensor([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 255]]
    ], dtype=torch.uint8)
    print("原始图片张量：\n", image_hwc)
    print("高度、宽度、通道形状：", list(image_hwc.shape))
    print("数据类型：", image_hwc.dtype)
    print("第一行第二列的绿色像素：", image_hwc[0, 1].tolist())
    print("全部红色分量：\n", image_hwc[:, :, 0])

    print("\n二、改变维度顺序，并加入批次维度")
    image_chw = image_hwc.permute(2, 0, 1)
    print("通道、高度、宽度形状：", list(image_chw.shape))
    print("CHW 顺序下读取同一个绿色像素：", image_chw[:, 0, 1].tolist())
    # 在最前面插入一个长度为 1 的维度，不会增加新像素。
    image_nchw = image_chw.unsqueeze(0)
    print("批次、通道、高度、宽度形状：", list(image_nchw.shape))
    repeated_batch = torch.stack([image_chw, image_chw], dim=0)
    print("重复两张教学图片组成批次：", list(repeated_batch.shape))

    print("\n三、类型转换、归一化和教学标准化")
    image_float = image_chw.float()
    print("仅转浮点数后的绿色像素：", image_float[:, 0, 1].tolist())
    print("转为浮点数后的类型：", image_float.dtype)
    image_01 = image_float / 255.0
    print("除以 255 后的绿色像素：", image_01[:, 0, 1].tolist())
    print("缩放后的最小值、最大值：", image_01.min().item(), image_01.max().item())
    # 以下均值和标准差仅为手算演示，不是任何预训练模型的通用配置。
    mean = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    image_standardized = (image_01 - mean) / std
    print("教学标准化后的绿色像素：", image_standardized[:, 0, 1].tolist())

    print("\n四、图像坐标和张量索引")
    horizontal = 1  # 横坐标，也就是列位置。
    vertical = 0  # 纵坐标，也就是行位置。
    print("图像坐标 (1, 0) 对应 HWC[0, 1]：", image_hwc[vertical, horizontal].tolist())

    print("\n五、物体框格式与交并比")
    box_a = torch.tensor([10., 10., 50., 50.])
    box_b = torch.tensor([30., 30., 70., 70.])
    print("框 A，xyxy：", box_a.tolist())
    print("框 B，xyxy：", box_b.tolist())
    print("框 A，中心横坐标、中心纵坐标、宽、高：", xyxy_to_cxcywh(box_a).tolist())
    print("框 A 面积：", (box_a[2:] - box_a[:2]).prod().item())
    print("框 B 面积：", (box_b[2:] - box_b[:2]).prod().item())
    intersection_size = (
        torch.minimum(box_a[2:], box_b[2:])
        - torch.maximum(box_a[:2], box_b[:2])
    ).clamp(min=0)
    print("交集宽、高：", intersection_size.tolist())
    print("交集面积：", intersection_size.prod().item())
    print("并集面积：", 1600.0 + 1600.0 - intersection_size.prod().item())
    print(f"框 A、B 的交并比：{single_box_iou(box_a, box_b).item():.6f}")
    print(f"框 A 和自身的交并比：{single_box_iou(box_a, box_a).item():.6f}")
    separated_box = torch.tensor([80., 80., 100., 100.])
    print(f"两个分离框的交并比：{single_box_iou(box_a, separated_box).item():.6f}")

    print("\n六、局部区域与卷积核的教学计算")
    patch = torch.tensor([[1., 2.], [4., 5.]])
    kernel = torch.tensor([[1., 0.], [0., -1.]])
    response = (patch * kernel).sum()
    print("局部区域：\n", patch)
    print("人为指定的卷积核：\n", kernel)
    print("对应元素相乘：\n", patch * kernel)
    print("相乘后求和的响应：", response.item())

    print("\n七、用张量形状表示物体特征")
    # 每行一个物体，每列一个特征分量；这些不是模型提取的真实特征。
    object_features = torch.tensor([
        [0.2, -0.4, 0.8, 0.1],  # 示意“人”的四维特征。
        [0.7, 0.3, -0.2, 0.5]   # 示意“杯子”的四维特征。
    ])
    print("两个物体，每个物体四个特征数：", list(object_features.shape))
    pair_feature = torch.cat([object_features[0], object_features[1]])
    print("拼接人的特征和杯子的特征：", pair_feature)
    print("拼接后特征长度：", pair_feature.numel())
    print("这些特征只演示数字排列，没有完成真实检测或关系预测。")

    print("\n独立练习：")
    print("1. 将四个像素改成黑、白、绿、红，预测三个颜色通道。")
    print("2. 在自己的文件中写出 HWC → CHW → 浮点数/255 → NCHW。")
    print("3. 用框 [20, 10, 80, 50] 手算中心宽高，再用函数核对。")
    print("4. 将框改成 [0, 0, 20, 20] 和 [10, 10, 30, 30]，手算交并比。")
    print("5. 将卷积核改成全 1，先预测局部响应，再运行。")


if __name__ == "__main__":
    main()
