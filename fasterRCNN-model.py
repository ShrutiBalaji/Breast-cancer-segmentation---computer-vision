import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.ops import box_iou

class MammographyDataset(data.Dataset):
    """Custom Dataset for loading mammography images"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load labels
        self.labels = np.load(os.path.join(data_dir, f'y_{split}.npy'))
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(data_dir) 
                           if f.startswith(f'X_{split}_') and f.endswith('.npy')]
        self.image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = np.load(img_path)
        
        # Convert to tensor and add channel dimension if needed
        image = torch.FloatTensor(image)
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        
        # Get label
        label = self.labels[idx]
        
        # Create target dict for Faster R-CNN
        boxes = torch.FloatTensor([[0, 0, image.shape[2], image.shape[1]]])  # Whole image box
        labels = torch.tensor([label], dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((1,), dtype=torch.int64)
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

def get_model(num_classes):
    """Get Faster R-CNN model with custom number of classes"""
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model





def train_one_epoch(model, optimizer, data_loader, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader, desc='Training'):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def compute_iou(pred_boxes, target_boxes):
    if pred_boxes.size(0) == 0 or target_boxes.size(0) == 0:
        return 0.0  # Return 0 if there are no boxes
    iou = box_iou(pred_boxes, target_boxes)  # Remove unsqueeze
    return iou.mean().item()  # Return the mean IoU for simplicity


def evaluate(model, data_loader, device):
    model.eval()  # Ensure model is in eval mode
    total_iou = 0
    count = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = [image.to(device) for image in images]
            outputs = model(images)  # Get predictions

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu()
                target_boxes = target['boxes'].cpu()

                # Compute IoU
                if len(pred_boxes) > 0 and len(target_boxes) > 0:
                    iou = compute_iou(pred_boxes, target_boxes)
                else:
                    iou = 0.0
                
                total_iou += iou
                count += 1

    mean_iou = total_iou / count if count > 0 else 0
    print(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou


def main():
    # Configuration
    DATA_DIR = '/Users/shrutibalaji/Downloads/vindr-mammo-master 2/preprocessed_data'
    NUM_CLASSES = 2  # Background + 1 class
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.005
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    transform = T.Compose([
        T.Lambda(lambda x: x / 255.0),  # Normalize to [0,1]
    ])
    
    train_dataset = MammographyDataset(DATA_DIR, 'train', transform)
    test_dataset = MammographyDataset(DATA_DIR, 'test', transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize model
    model = get_model(NUM_CLASSES)
    model.to(device)
    
    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    # Training loop
    best_metric = float('-inf')  # Track best IoU (or any chosen metric)
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        metric = evaluate(model, test_loader, device)
        print(f"Validation Metric (IoU): {metric:.4f}")
        
        # Save best model
        if metric > best_metric:
            best_metric = metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': best_metric,
            }, 'best_model.pth')
            print("Saved best model")


if __name__ == "__main__":
    main()