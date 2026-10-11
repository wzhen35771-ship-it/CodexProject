"""第九课：训练小模型，保存参数，加载后核对预测。"""
from pathlib import Path

import torch

torch.manual_seed(0)
x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
target = 3 * x + 1
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
criterion = torch.nn.MSELoss()

model.train()
for step in range(300):
    optimizer.zero_grad()
    loss = criterion(model(x), target)
    loss.backward()
    optimizer.step()

# 权重文件始终放在本脚本旁的输出目录，方便查找。
output_directory = Path(__file__).resolve().parent / "outputs"
output_directory.mkdir(exist_ok=True)
weight_path = output_directory / "09_保存与加载_权重.pt"
torch.save(model.state_dict(), weight_path)
print("参数已保存到：", weight_path)
print("保存的参数名称：", list(model.state_dict().keys()))

# 创建结构相同的新模型，再把保存的参数装进去。
loaded_model = torch.nn.Linear(1, 1)
state = torch.load(weight_path, map_location="cpu", weights_only=True)
loaded_model.load_state_dict(state)

# 评估模式和暂停求导各有职责，两者一起用于这里的预测。
model.eval()
loaded_model.eval()
test_x = torch.tensor([[4.0], [5.0]])
with torch.no_grad():
    original_prediction = model(test_x)
    loaded_prediction = loaded_model(test_x)

print("测试输入：", test_x.squeeze(1).tolist())
print("原模型预测：", original_prediction.squeeze(1).tolist())
print("加载后预测：", loaded_prediction.squeeze(1).tolist())
print("预测结果是否一致：", torch.allclose(original_prediction, loaded_prediction))
print("练习：把目标规律改为 2 * x - 1，重新训练并核对保存、加载后的预测。")
