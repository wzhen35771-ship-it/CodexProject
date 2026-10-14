import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from importlib import import_module

SmallClassifier = import_module('05_models_example').SmallClassifier
make_data = import_module('06_data_example').make_data

def train_model(verbose=True):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    features, labels = make_data()
    loader = DataLoader(TensorDataset(features, labels), batch_size=4, shuffle=True,
                        generator=torch.Generator().manual_seed(7))
    model = SmallClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    updates = 0
    for epoch in range(1, 201):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            updates += 1
        if verbose and (epoch == 1 or epoch % 50 == 0):
            print('轮数：', epoch, '平均训练损失：', total_loss / len(labels))
    model.eval()
    with torch.no_grad():
        logits = model(features)
        accuracy = (logits.argmax(1) == labels).float().mean().item()
        final_loss = criterion(logits, labels).item()
    if verbose:
        print('训练准确率：', accuracy, '最终损失：', final_loss, '更新次数：', updates)
    assert accuracy == 1.0 and final_loss < 0.1
    assert updates == 600
    return model

if __name__ == '__main__':
    train_model()
