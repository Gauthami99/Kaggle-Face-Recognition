# Imports
import face_recognition
from PIL import Image, ImageOps
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install face_recognition library
!pip install face_recognition

# Function to convert images to grayscale and save them
def greyscale_image(source_folder, csv_file, target_size=(224, 224), save_dir='/content/drive/MyDrive/MLChallenge/greyscale_images'):
    # Read image labels from CSV file
    image_labels = pd.read_csv(csv_file)
    # Iterate over each image in the CSV
    for index, row in image_labels.iterrows():
        image_name = row[1]
        image_path = os.path.join(source_folder, image_name)
        try:
            # Open image and detect faces
            image = Image.open(image_path)
            image_array = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image_array)

            # Convert image to grayscale
            image = image.convert('L')

            # If no faces detected, crop center of image
            if len(face_locations) == 0:
                w, h = image.size
                start_x = max(0, w // 2 - target_size[0] // 2)
                start_y = max(0, h // 2 - target_size[1] // 2)
                end_x = start_x + target_size[0]
                end_y = start_y + target_size[1]
                patch = image.crop((start_x, start_y, end_x, end_y))
            else:
                top, right, bottom, left = face_locations[0]
                face = image.crop((left, top, right, bottom))
                # Resize and center crop the face
                patch = ImageOps.fit(face, target_size, method=Image.ANTIALIAS, centering=(0.5, 0.5))

            # Convert the patch to grayscale
            patch = patch.convert('L')

            # Save the processed image
            save_path = os.path.join(save_dir, f"processed_{image_name}")
            patch.save(save_path)
            print(f"Processed image saved to {save_path}")

        except Exception as e:
            # Handle exceptions by saving a blank image
            print(f"Error for image {image_name}: {e}.")
            blank_image = Image.new('L', target_size, (0))
            save_path = os.path.join(save_dir, f"processed_{image_name}")
            blank_image.save(save_path)

# Source folder and CSV file containing image labels
source_folder = '/content/drive/MyDrive/train'
csv_file = '/content/drive/MyDrive/MLChallenge/train.csv'
# Call the function to convert images to grayscale
greyscale_image(source_folder, csv_file)

# Dataset class for handling image data
class Faces(Dataset):
    def __init__(self, image_dir, csv_file, category_file, transform=None):
        self.image_labels = pd.read_csv(csv_file)
        self.categories = pd.read_csv(category_file)['Category'].tolist()
        self.category_to_index = {category: i for i, category in enumerate(self.categories)}
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        # Get the path of the processed image
        image_path = os.path.join(self.image_dir, f"processed_{self.image_labels.iloc[idx, 0]}.jpg")
        # Open the image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            # Apply transformations if specified
            image = self.transform(image)

        # Get the label name and its index
        label_name = self.image_labels.iloc[idx]['Category']
        label_index = self.category_to_index[label_name]

        return image, label_index

# Define the ResNet50 model
trained_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_features = trained_model.fc.in_features
# Replace the fully connected layer for custom classification
trained_model.fc = nn.Linear(num_features, 100)

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create an instance of the dataset
dataset = Faces(image_dir='/content/drive/MyDrive/MLChallenge/tprocessed_images',
                csv_file='/content/drive/MyDrive/MLChallenge/train.csv',
                category_file='/content/drive/MyDrive/MLChallenge/category.csv',
                transform=transform
                )
total = len(dataset)
train = int(0.8 * total)
valid = total - train
# Split the dataset into training and validation sets
training_dataset, validation_dataset = random_split(dataset, [train, valid])

# Create data loaders for training and validation
training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)

# Check if GPU is available, and move the model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(trained_model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training
    trained_model.train()
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = trained_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    trained_model.eval()
    correct_valid = 0
    total_valid = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()

    valid_accuracy = 100 * correct_valid / total_valid
    print(f'Epoch {epoch+1}, Validation Accuracy: {valid_accuracy}%')

# Save the trained model
torch.save(trained_model.state_dict(), 'extractor.pth')



# TESTING PHASE

# Path to the trained model
feature_extractor_path = 'extractor.pth'

# Initialize the model and load the trained weights
trained_model.load_state_dict(torch.load(feature_extractor_path, map_location=device))
trained_model.to(device)
trained_model.eval()

# Dataset class for processed images
class TestProcessDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        # Get list of image filenames in the directory
        self.image_filenames = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.directory, img_name)
        # Open and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        return image, img_name

# Define image transformations
image_transform = transforms.Compose([  
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Read category data
category_data = pd.read_csv('/content/drive/MyDrive/MLChallenge/category.csv')
# Convert category names to dictionary for mapping
category_mapping = category_data['Category'].to_dict()

# Function to map prediction index to category name
def map_prediction_to_name(prediction):
    return category_mapping.get(prediction, "Unknown")

# Directory containing processed test images
testing_data_images_dir = '/content/drive/MyDrive/MLChallenge/testing_data_processed_images'
# Create dataset instance
testing_dataset = TestProcessDataset(directory=testing_data_images_dir, transform=image_transform) 
# Create DataLoader for the test dataset
testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)  

# List to store predictions
prediction_list = []
# Iterate over test DataLoader
for inputs, image_names in testing_dataloader: 
    inputs = inputs.to(device)
    with torch.no_grad():
        # Forward pass through the model
        outputs = trained_model(inputs)
        _, predicted = torch.max(outputs, 1)
        # Convert predictions to numpy array
        predicted = predicted.cpu().numpy()
        # Associate predicted labels with image names
        for img_name, prediction in zip(image_names, predicted):
            prediction_list.append((img_name, map_prediction_to_name(prediction)))

# Path to save prediction file
prediction_file = '/content/drive/MyDrive/MLChallenge/Submission.csv'  
with open(prediction_file, 'w') as f:
    f.write('Id,Category\n')
    # Write predictions to file
    for img_name, pred_name in prediction_list:
        f.write(f"{img_name},{pred_name}\n")
print(f"Predictions saved to {prediction_file}")
