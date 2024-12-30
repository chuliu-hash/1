import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from net import ResNet
from net import BasicBlock


# 保存检查点
def save_checkpoint(model, optimizer, epoch):
    checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_filename)
    print(f"Checkpoint saved: {checkpoint_filename}")


# 加载模型检查点
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        print("No checkpoint found.")


if __name__ == "__main__":
    # 超参数
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001

    # 数据预处理和增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomCrop(32, padding=4),  # 随机裁剪，加上填充
        transforms.Resize((224, 224)),  # 将图像调整为224x224
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])

    # 下载和加载 CIFAR-10 数据集
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = ResNet(BasicBlock)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    load_checkpoint(model, optimizer, 'checkpoint_epoch_0.pth')  # 可选的最新检查点

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 保存检查点
        save_checkpoint(model, optimizer, epoch + 1)  # 保存每个 epoch 的检查点
