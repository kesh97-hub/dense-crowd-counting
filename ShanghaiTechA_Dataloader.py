import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageOps
import scipy.io
from scipy.ndimage import gaussian_filter

class ShanghaiTechA(Dataset):
    def __init__(self, image_path, annotation_path, target_size=(224, 224), sigma=4, transform=None):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.target_size = target_size
        self.sigma = sigma
        self.transform = transform
        self.images, self.annotations = self.load_data()
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mat_path = self.annotations[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        mat = scipy.io.loadmat(mat_path)
        annPoints = mat['image_info'][0, 0][0, 0][0]  # Adjust to match the structure of ShanghaiTechA
        image, density_map, reinforcement_map = self.preprocess_image_and_annotations(image, annPoints)
        
        if self.transform:
            image, density_map, reinforcement_map = self.transform(image, density_map, reinforcement_map)
        
        image = self.image_transform(image)
        density_map = torch.from_numpy(density_map).unsqueeze(0)  # Add channel dimension
        reinforcement_map = torch.from_numpy(reinforcement_map).unsqueeze(0)  # Add channel dimension
        return image, density_map, reinforcement_map

    def load_data(self):
        images = []
        annotations = []
        for filename in os.listdir(self.image_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(self.image_path, filename)
                mat_path = os.path.join(self.annotation_path, f"GT_{os.path.splitext(filename)[0]}.mat")
                if os.path.exists(mat_path):
                    images.append(img_path)
                    annotations.append(mat_path)
        return images, annotations

    def preprocess_image_and_annotations(self, image, annPoints):
        original_width, original_height = image.size
        target_height, target_width = self.target_size
        resized_image = image.resize((target_width, target_height), Image.LANCZOS)

        scale_x = target_width / original_width
        scale_y = target_height / original_height

        annPoints = np.array(annPoints)
        adjusted_annotations = annPoints * [scale_x, scale_y]
        
        density_map = np.zeros((target_height, target_width), dtype=np.float32)
        reinforcement_map = np.zeros((target_height, target_width), dtype=np.float32)
        for point in adjusted_annotations:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < target_width and 0 <= y < target_height:
                density_map[y, x] += 1
                reinforcement_map[y, x] = 1
                
        density_map = gaussian_filter(density_map, sigma=self.sigma)
        reinforcement_map = gaussian_filter(reinforcement_map, sigma=self.sigma * 2)
        reinforcement_map = (reinforcement_map > 0.001).astype(np.float32)
        return resized_image, density_map, reinforcement_map

    
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, density_map, reinforcement_map):
        if random.random() < self.p:
            image = ImageOps.mirror(image)
            density_map = np.fliplr(density_map).copy()
            reinforcement_map = np.fliplr(reinforcement_map).copy()
        return image, density_map, reinforcement_map
    


def prepare_dataloaders(train_image_path, train_annotation_path, test_image_path, test_annotation_path, val_split=0.2):
    transform = RandomHorizontalFlip(p=0.5)
    full_train_dataset = ShanghaiTechA(train_image_path, train_annotation_path, transform=transform)
    total_train_size = len(full_train_dataset)
    val_size = int(total_train_size * val_split)
    train_size = total_train_size - val_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    test_dataset = ShanghaiTechA(test_image_path, test_annotation_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader