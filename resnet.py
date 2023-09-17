import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
# 设置随机种子
torch.manual_seed(42)

# 数据路径
train_datadir = r'./train'
test_datadir = r'./test'
# 图像预处理和数据增强
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载数据集
train_data = datasets.ImageFolder(train_datadir,transform=transform)

test_data = datasets.ImageFolder(test_datadir,transform=transform)

# 创建数据加载器
batch_size = 20
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

# 定义ResNet模型
model = torchvision.models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 二分类任务，输出2个类别

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
model.to(device)

for epoch in range(num_epochs):
    train_loss = 0.0
    test_loss = 0.0

    # 训练阶段
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    correct = 0
    total = 0
    # 验证阶段
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)

    accuracy = 100 * correct / total
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'resnet_model.pth')
