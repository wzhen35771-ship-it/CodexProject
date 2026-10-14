import torch

image = torch.arange(18).reshape(2, 3, 3)  # H=2, W=3, C=3
channels = image.permute(2, 0, 1)
batch = channels.unsqueeze(0)
print('HWC：', image.shape)
print('CHW：', channels.shape)
print('NCHW：', batch.shape)
print('原像素与变换后像素：', image[1, 2, 0], channels[0, 1, 2])
assert image[1, 2, 0] == batch[0, 0, 1, 2]
print('展平：', batch.reshape(1, -1).shape)
