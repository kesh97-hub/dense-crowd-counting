import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
import pytorch_ssim

class MSE_BCE_Loss(nn.Module):
    def __init__(self, alpha=1000, beta=10):
        super(MSE_BCE_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, 
                generated_density_maps, generated_reinforcement_maps, 
                target_density_maps, target_reinforcement_maps
               ):
        density_map_loss = self.mse_loss(generated_density_maps, target_density_maps)
        reinforcement_map_loss = self.bce_loss(generated_reinforcement_maps, target_reinforcement_maps)
        
        total_loss = self.alpha * density_map_loss + self.beta * reinforcement_map_loss
        return total_loss

    
class MAE_BCE_Loss(nn.Module):
    def __init__(self, alpha=1000, beta=10):
        super(MAE_BCE_Loss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, 
                generated_density_maps, generated_reinforcement_maps, 
                target_density_maps, target_reinforcement_maps
               ):
        density_map_loss = self.mae_loss(generated_density_maps, target_density_maps)
        reinforcement_map_loss = self.bce_loss(generated_reinforcement_maps, target_reinforcement_maps)
        total_loss = self.alpha * density_map_loss + self.beta * reinforcement_map_loss
        return total_loss

    
def calculate_density_map_metrics(preds, targets):
    mae = torch.mean(torch.abs(preds - targets))
    mse = torch.mean((preds - targets) ** 2)
    rmse = torch.sqrt(mse)
    
    return mae.item(), mse.item(), rmse.item()


def calculate_count_metrics(pred, target):
    predicted_count = pred.sum(dim=[1,2,3])
    target_count = target.sum(dim=[1,2,3])
    
    mae = torch.mean(torch.abs(predicted_count - target_count))
    mse = torch.mean((predicted_count - target_count) ** 2)
    rmse = torch.sqrt(mse)
    
    return mae.item(), mse.item(), rmse.item()


def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()
    

def visualize_density_maps(images, ground_truth, prediction, sigma=2):
    batch_size = images.size(0)
    restore_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                             std=[1/0.229, 1/0.224, 1/0.225])
    ])
    for i in range(batch_size):
        image = restore_transform(images[i]).permute(1,2,0).cpu().numpy()
        gt_density_map = ground_truth[i].cpu().squeeze(0).numpy()
        pred_density_map = prediction[i].cpu().squeeze(0).numpy()
        
        pred_density_map = gaussian_filter(pred_density_map, sigma=sigma)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(gt_density_map, cmap='jet')
        axs[1].set_title(f'Ground Truth Density Map (Count: {int(np.sum(gt_density_map))})')
        axs[1].axis('off')

        im = axs[2].imshow(pred_density_map, cmap='jet')
        axs[2].set_title(f'Predicted Density Map (Count: {int(np.sum(pred_density_map))})')
        axs[2].axis('off')

        fig.colorbar(im, ax=axs, orientation='horizontal', fraction=.1)
        plt.show()

        