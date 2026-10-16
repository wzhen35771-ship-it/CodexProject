import torch
from torch import nn

class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3))

    def forward(self, features):
        return self.layers(features)

def main():
    torch.manual_seed(7)
    model = SmallClassifier()
    features = torch.tensor([[-0.6, 0.], [0.6, 0.], [0., -0.6]])
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    logits = model(features)
    loss = nn.CrossEntropyLoss()(logits, labels)
    print('输入、输出、标签形状：', features.shape, logits.shape, labels.shape)
    print('logits：', logits, '损失：', loss.item())
    for name, parameter in model.named_parameters():
        print(name, parameter.shape)
    print('参数总数：', sum(p.numel() for p in model.parameters()))
    assert logits.shape == (3, 3)

if __name__ == '__main__':
    main()
