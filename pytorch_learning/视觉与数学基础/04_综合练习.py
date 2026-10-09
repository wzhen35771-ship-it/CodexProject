"""综合练习：串起图片形状、给定物体框、示例特征、关系损失与一次更新。"""
import torch

torch.manual_seed(7)

# 为练习形状而创建的全黑图片；下面的框和特征都是人工指定。
# 本程序没有从图片检测物体，不代表训练好的视觉识别系统。
image_hwc = torch.zeros((100, 100, 3), dtype=torch.uint8)
image_nchw = (image_hwc.float() / 255).permute(2, 0, 1).unsqueeze(0)
print("一、图片形状：", list(image_hwc.shape), "→", list(image_nchw.shape))

boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [30.0, 30.0, 70.0, 70.0]])
intersection_size = (torch.minimum(boxes[0, 2:], boxes[1, 2:])
                     - torch.maximum(boxes[0, :2], boxes[1, :2])).clamp(min=0)
intersection_area = intersection_size.prod()
areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
iou = intersection_area / (areas.sum() - intersection_area)
print("二、给定框的交并比：", round(iou.item(), 6))

# 两个物体各有三个教学特征，拼接成这个物体对的六个特征。
object_features = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
pair_features = torch.cat([object_features[0], object_features[1]]).unsqueeze(0)
print("三、物体特征形状：", list(object_features.shape))
print("物体对特征形状：", list(pair_features.shape))

# 三个独立关系是否成立，使用多标签二元损失。
model = torch.nn.Linear(6, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
targets = torch.tensor([[1.0, 0.0, 1.0]])
mask = torch.tensor([[0.0, 1.0, 0.0]])  # 第二项屏蔽。
keep = 1 - mask

optimizer.zero_grad()
scores = model(pair_features)
individual_losses = criterion(scores, targets)
loss = (individual_losses * keep).sum() / keep.sum().clamp(min=1)
loss.backward()

print("四、关系原始得分形状：", list(scores.shape))
print("目标与掩码形状：", list(targets.shape), list(mask.shape))
print("逐项损失：", individual_losses.detach().tolist())
print("本次有效损失：", round(loss.item(), 6))
print("分类层权重梯度形状：", list(model.weight.grad.shape))
print("第二个输出对应的权重行梯度：", model.weight.grad[1].tolist())

optimizer.step()
with torch.no_grad():
    new_scores = model(pair_features)
    new_losses = criterion(new_scores, targets)
    new_loss = (new_losses * keep).sum() / keep.sum().clamp(min=1)

print("五、更新后的有效损失：", round(new_loss.item(), 6))
print("本例一次更新后损失是否降低：", new_loss.item() < loss.item())
print("说明：这里只训练一个小关系分类层，没有训练图像特征提取器。")
