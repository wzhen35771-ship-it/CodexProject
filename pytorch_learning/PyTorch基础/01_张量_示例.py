"""人工张量、图片维度和广播。全部在 CPU 上运行，无需下载数据。"""

import torch


def main():
    # 三行四列。先在纸上算出索引和求和的值，再运行核对。
    values = torch.tensor(
        [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]],
        dtype=torch.float32,
    )
    print("PyTorch 版本：", torch.__version__)
    print("原张量：\n", values)
    print("形状与类型：", tuple(values.shape), values.dtype)
    print("第二行：", values[1], "形状：", tuple(values[1].shape))
    print("第二行切片：", values[1:2], "形状：", tuple(values[1:2].shape))
    print("最后一列：", values[:, -1])
    print("第一行第二列：", values[0, 1].item())
    print("dim=0 求和：", values.sum(dim=0))
    print("dim=1 求和：", values.sum(dim=1))
    print("dim=1 保留轴：", values.sum(dim=1, keepdim=True))
    assert torch.equal(values.sum(dim=0), torch.tensor([12, 15, 18, 21]))
    assert torch.equal(values.sum(dim=1), torch.tensor([10, 22, 34]))

    # [4] 按最后一轴广播到 [3, 4]，每行都减去同一组数。
    offset = torch.tensor([1.0, 2.0, 3.0, 4.0])
    centered = values - offset
    print("按列广播相减：\n", centered)
    assert torch.equal(centered, torch.tensor([[0.0] * 4, [3.0] * 4, [6.0] * 4]))

    # reshape 改变形状，总元素数保持为 12。
    reshaped = values.reshape(2, -1)
    print("reshape(2, -1)：\n", reshaped, "形状：", tuple(reshaped.shape))
    assert reshaped.shape == (2, 6)

    # HWC：高 2、宽 3、每个像素有 R/G/B 三个值。
    image_hwc = torch.tensor(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        ],
        dtype=torch.float32,
    )
    image_chw = image_hwc.permute(2, 0, 1)
    single_batch = image_chw.unsqueeze(0)
    image_batch = torch.stack([image_chw, image_chw], dim=0)
    print("HWC：", tuple(image_hwc.shape))
    print("CHW：", tuple(image_chw.shape))
    print("单图 batch：", tuple(single_batch.shape))
    print("双图 batch：", tuple(image_batch.shape))
    print("同一个像素 HWC：", image_hwc[1, 2, :])
    print("同一个像素 CHW：", image_chw[:, 1, 2])
    print("R 通道：\n", image_chw[0])
    assert torch.equal(image_hwc[1, 2, :], image_chw[:, 1, 2])

    # 同样的目标形状，不代表图片里的元素排列相同。
    wrong_chw = image_hwc.reshape(3, 2, 3)
    print("正确交换轴的 R[0,1]：", image_chw[0, 0, 1].item())
    print("仅改形状的同一位置：", wrong_chw[0, 0, 1].item())
    assert image_chw[0, 0, 1].item() == 40.0
    assert wrong_chw[0, 0, 1].item() == 20.0

    # 教学归一化参数。真实模型需要遵循该模型的预处理配置。
    image_float = image_batch / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
    normalized = (image_float - mean) / std
    print("归一化输出形状：", tuple(normalized.shape))
    print("第一张图第一个像素：", normalized[0, :, 0, 0])
    assert normalized.shape == image_batch.shape
    print("形状与数值检查通过。接下来脱离示例完成过关任务。")


if __name__ == "__main__":
    main()
