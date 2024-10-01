import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

# Define the transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the pre-trained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Modify the last layer for binary classification
model.load_state_dict(torch.load('flower_model.pth'))  # Load the trained model
model.eval()
def classify_image(img_path):
    try:
        # Load and preprocess the image
        print(f"Loading image from {img_path}")
        img = Image.open(img_path)
        img_tensor = preprocess(img).unsqueeze(0)
        print("Image preprocessed.")
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        print(f"Prediction: {predicted.item()}")
        
        # Display the image and prediction
        plt.imshow(img)
        plt.axis('off')
        
        # Map predicted class index to class names
        # Assuming 0 is 'flower' and 1 is 'not flower'
        class_name = 'flower' if predicted.item() == 0 else 'not flower'
        plt.title(class_name)
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
classify_image(r'C:\Users\palwa\Desktop\flower\flower\flower\FD.3.jpg')


