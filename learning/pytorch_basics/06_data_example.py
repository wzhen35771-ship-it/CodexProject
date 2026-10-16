import torch
from torch.utils.data import DataLoader, TensorDataset

def make_data():
    features = torch.tensor([
        [-.8, 0.], [-.6, .1], [-.7, -.1], [-.5, 0.],
        [.8, 0.], [.6, .1], [.7, -.1], [.5, 0.],
        [0., -.8], [.1, -.6], [-.1, -.7], [0., -.5]])
    labels = torch.tensor([0]*4 + [1]*4 + [2]*4, dtype=torch.long)
    return features, labels

def main():
    features, labels = make_data()
    loader = DataLoader(TensorDataset(features, labels), batch_size=5, shuffle=False)
    for index, (x, y) in enumerate(loader):
        print('批次：', index, '形状：', x.shape, y.shape, '标签：', y.tolist())
    assert len(loader) == 3

if __name__ == '__main__':
    main()
