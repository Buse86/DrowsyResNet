import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
# Параметры
data_dir = 'train_classes'  # Укажите путь к вашему датасету
batch_size = 32
num_classes = 10
num_epochs = 10
learning_rate = 0.001
torch.device('cuda')
# Аугментация и нормализация изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Изменение яркости и контрастности
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка данных
train_dataset = datasets.ImageFolder(data_dir, transform)


from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)


#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Использование предобученной модели ResNet
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Добавленный слой Dropout
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = CustomResNet()

#model = models.resnet34(pretrained=True)





#model.fc = nn.Linear(model.fc.in_features, num_classes)  # Изменяем последний слой
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Обучение модели
model.train()
min_loss = 1
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
    if loss.item() < min_loss:
        min_loss = loss.item()
        torch.save(model.state_dict(), 'best_ animal_classifier.pth')
    writer.add_scalar(f'Loss/train', loss.item(), epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
writer.close()
# Сохранение модели
torch.save(model.state_dict(), 'animal_classifier.pth')