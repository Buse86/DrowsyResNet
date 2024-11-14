import cv2
import numpy as np
from joblib import load
import keyboard
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 250)


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Добавленный слой Dropout
            nn.Linear(self.model.fc.in_features, 2)
        )

    def forward(self, x):
        return self.model(x)


model = CustomResNet()

model.load_state_dict(torch.load('best_drowsy_classifier.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    # Изменение яркости и контрастности
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
verd = ''

while True:
    _, image = cap.read()

    image_orig = image
    # print(_)
    if keyboard.is_pressed('q'):
        cv2.imwrite(f"LastFrame.jpg", image)
        image1 = Image.open('LastFrame.jpg')
        image_tensor = preprocess(image1)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu()
            top_prob, top_class = torch.topk(probabilities, 1)
            top_prob = top_prob.item()
            top_class = top_class.item()
            verd = top_class
            print(top_class)

    cv2.putText(image_orig, f'status: {verd}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("output", image)

    k = cv2.waitKey(5) & 0xFF

    if keyboard.is_pressed('5'):
        break

    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
keyboard.wait('+')

