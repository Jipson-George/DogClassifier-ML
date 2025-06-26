# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class DogDataset(Dataset):
    """Custom dataset for dog classification"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class SimpleCNN(nn.Module):
    """Simple but effective CNN for dog classification"""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # Input size: 128 * 9 * 9 (after 4 pooling operations: 150/2/2/2/2 = 9.375 ≈ 9)
        self.fc1 = nn.Linear(128 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(self.relu(self.conv1(x)))  # 32 x 75 x 75
        x = self.pool(self.relu(self.conv2(x)))  # 64 x 37 x 37
        x = self.pool(self.relu(self.conv3(x)))  # 128 x 18 x 18
        x = self.pool(self.relu(self.conv4(x)))  # 128 x 9 x 9
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 9 * 9)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        
        return x

class DogClassifier:
    def __init__(self, img_size=150, device=None):
        self.img_size = img_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        print(f"Using device: {self.device}")
    
    def load_and_preprocess_data(self, data_dir):
        """Load images from directory structure"""
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        
        # Load dog images (label = 1)
        dog_dir = os.path.join(data_dir, 'dogs')
        if os.path.exists(dog_dir):
            for filename in os.listdir(dog_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(dog_dir, filename)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((self.img_size, self.img_size))
                        images.append(img)
                        labels.append(1)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        # Load not-dog images (label = 0)
        not_dog_dir = os.path.join(data_dir, 'not_dogs')
        if os.path.exists(not_dog_dir):
            for filename in os.listdir(not_dog_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(not_dog_dir, filename)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((self.img_size, self.img_size))
                        images.append(img)
                        labels.append(0)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(images)} images total")
        print(f"Dogs: {sum(labels)}, Not dogs: {len(labels) - sum(labels)}")
        
        return images, labels
    
    def get_transforms(self, is_training=True):
        """Get data transforms for training and validation"""
        if is_training:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def train_epoch(self, dataloader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images).squeeze()
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def train(self, data_dir, epochs=25, batch_size=16, learning_rate=0.001, validation_split=0.2):
        """Train the model"""
        # Load data
        images, labels = self.load_and_preprocess_data(data_dir)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create datasets
        train_dataset = DogDataset(X_train, y_train, transform=self.get_transforms(True))
        val_dataset = DogDataset(X_val, y_val, transform=self.get_transforms(False))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = SimpleCNN().to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        print("Starting training...")
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print('-' * 50)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def plot_training_history(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history['train_acc'], label='Training Accuracy')
        ax1.plot(self.history['val_acc'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history['train_loss'], label='Training Loss')
        ax2.plot(self.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def save_model(self, filepath='dog_classifier_model.pth'):
        """Save the trained model"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'img_size': self.img_size,
            }, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, filepath='dog_classifier_model.pth'):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        if 'img_size' in checkpoint:
            self.img_size = checkpoint['img_size']
        
        print(f"Model loaded from {filepath}")
    
    def predict(self, image_path):
        """Predict if an image contains a dog"""
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        
        # Apply same normalization as training
        transform = self.get_transforms(False)
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(img_tensor).item()
        
        is_dog = prediction > 0.5
        confidence = prediction if is_dog else 1 - prediction
        
        return {
            'is_dog': is_dog,
            'confidence': float(confidence),
            'raw_prediction': float(prediction)
        }

def main():
    # Initialize classifier
    classifier = DogClassifier()
    
    # Create data directory structure if it doesn't exist
    os.makedirs('data/dogs', exist_ok=True)
    os.makedirs('data/not_dogs', exist_ok=True)
    
    print("Dog Classifier Training Script (PyTorch)")
    print("=======================================")
    print("Expected directory structure:")
    print("data/")
    print("  dogs/")
    print("    dog1.jpg, dog2.jpg, ...")
    print("  not_dogs/")
    print("    cat1.jpg, other1.jpg, ...")
    print()
    
    # Check if data exists
    data_dir = 'data'
    if not os.path.exists(os.path.join(data_dir, 'dogs')) or not os.path.exists(os.path.join(data_dir, 'not_dogs')):
        print("Please create the data directory structure and add your images.")
        return
    
    # Train the model
    try:
        history = classifier.train(data_dir, epochs=25, batch_size=16)
        
        # Save the model
        classifier.save_model('dog_classifier_model.pth')
        
        print("\nTraining completed!")
        print("Model saved as 'dog_classifier_model.pth'")
        print("Training history plot saved as 'training_history.png'")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()