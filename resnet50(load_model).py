import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


Net = models.resnet50()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Net.to(device)
Net.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = CIFAR10(root="", train=True, transform=transform, download=False)
test_set = CIFAR10(root="", train=False, transform=transform, download=False)
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
test_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
Net.load_state_dict(torch.load('model/model30.pt'))


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x = x.to(device)
            y = y.to(device)
            outputs = net(x)
            preds = torch.argmax(outputs, dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total


print("model accuracy:", evaluate(test_dataloader, Net))
