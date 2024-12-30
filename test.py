import torch
from PIL import Image
import torchvision.transforms as transforms
from eval import load_model


def classify_image(img_path):
    # 数据预处理和增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为224x224
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])
    # 加载模型
    model = load_model('checkpoint_epoch_20.pth')  # 可选的最新检查点
    device = torch.device('cpu')
    model.to(device)
    model.eval()  # 设置为评估模式

    with torch.no_grad():
        img = Image.open(img_path)
        img = transform(img).unsqueeze(0).to(device)  # 添加批次维度并移动到设备
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()




if __name__ =="__main__":
    image_path = 'bird.jpg'
    predicted_class = classify_image(image_path)
    print(f'Predicted class for the image: {predicted_class}')

