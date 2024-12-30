import os
from torch.utils.data import DataLoader
import torchvision
from net import ResNet, BasicBlock
import torch
import torchvision.transforms as transforms


# 加载模型
def load_model(checkpoint_path):
    model = ResNet(BasicBlock)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    else:
        print("No checkpoint found.")
        exit()


if __name__ == "__main__":
    # 数据预处理和增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为224x224
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])
    # 数据预处理和增强

    # 超参数
    batch_size = 64
    # 下载和加载 CIFAR-10 数据集
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = load_model('checkpoint_epoch_20.pth')  # 可选的最新检查点

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')
