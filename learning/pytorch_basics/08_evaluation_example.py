from pathlib import Path
from importlib import import_module
import torch

SmallClassifier = import_module('05_models_example').SmallClassifier
train_model = import_module('07_training_example').train_model

def main():
    model = train_model(verbose=False)
    features = torch.tensor([[-.9, .05], [-.4, -.05], [.9, .05], [.4, -.05],
                             [.05, -.9], [-.05, -.4]])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    model.eval()
    with torch.no_grad():
        before = model(features)
        accuracy = (before.argmax(1) == labels).float().mean().item()
    print('人工评估样本准确率：', accuracy)
    output = Path(__file__).resolve().parent / 'outputs'
    output.mkdir(exist_ok=True)
    path = output / '08_classifier_weights.pt'
    torch.save(model.state_dict(), path)
    loaded = SmallClassifier()
    loaded.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    loaded.eval()
    with torch.no_grad():
        after = loaded(features)
    torch.testing.assert_close(before, after)
    print('加载前后 logits 一致；权重路径：', path)

if __name__ == '__main__':
    main()
