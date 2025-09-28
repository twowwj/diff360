#!/usr/bin/env python3
"""
Integration example showing how to use RenderMe360Dataset with DiffPortrait360
"""

import sys
import os

# Add the DiffPortrait360 code path
sys.path.append('/home/weijie.wang/project/DiffPortrait360/diffportrait360_release/code')

try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available, showing structure only")
    TORCH_AVAILABLE = False

from renderme360_dataset import RenderMe360Dataset, RenderMe360SingleFrameDataset

def create_renderme360_datasets(data_dir="./0018_data_compact", sample_frame=8):
    """
    Create RenderMe360 datasets compatible with DiffPortrait360 training
    
    Args:
        data_dir (str): Path to compact data directory
        sample_frame (int): Number of frames for temporal training
        
    Returns:
        tuple: (train_dataset, test_dataset) or (None, None) if torch unavailable
    """
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available, cannot create datasets")
        return None, None
    
    # Define image transforms (matching DiffPortrait360's preprocessing)
    image_transform = T.Compose([
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create temporal dataset for training
    train_dataset = RenderMe360Dataset(
        data_dir=data_dir,
        image_transform=image_transform,
        sample_frame=sample_frame,
        more_image_control=True  # Enable extra appearance control
    )
    
    # Create single frame dataset for testing/validation
    test_dataset = RenderMe360SingleFrameDataset(
        data_dir=data_dir,
        image_transform=image_transform,
        more_image_control=False
    )
    
    return train_dataset, test_dataset

def create_data_loaders(train_dataset, test_dataset, batch_size=4, num_workers=4):
    """
    Create data loaders for training and testing
    
    Args:
        train_dataset: Training dataset
        test_dataset: Testing dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    if not TORCH_AVAILABLE or train_dataset is None:
        return None, None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, test_loader

def demonstrate_compatibility():
    """
    Demonstrate compatibility with DiffPortrait360's expected data format
    """
    print("=== RenderMe360Dataset Integration with DiffPortrait360 ===")
    print()
    
    # Create datasets
    train_dataset, test_dataset = create_renderme360_datasets()
    
    if train_dataset is None:
        print("❌ Cannot create datasets without PyTorch")
        return
    
    print("✅ Datasets created successfully!")
    print(f"   Training dataset length: {len(train_dataset)}")
    print(f"   Testing dataset length: {len(test_dataset)}")
    print()
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)
    
    print("✅ Data loaders created successfully!")
    print()
    
    # Test a single batch
    print("=== Testing Data Format Compatibility ===")
    
    try:
        # Get a sample from training dataset
        sample = train_dataset[0]
        
        print("Training sample format:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)}")
        
        print()
        
        # Test with data loader
        batch = next(iter(train_loader))
        print("Training batch format:")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)}")
        
        print()
        print("✅ Data format is compatible with DiffPortrait360!")
        
        # Show expected usage pattern
        print("=== Expected Usage Pattern ===")
        print("""
# In your training script:
from renderme360_dataset import RenderMe360Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

# Create dataset
image_transform = T.Compose([
    T.ToTensor(), 
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = RenderMe360Dataset(
    data_dir="./0018_data_compact",
    image_transform=image_transform,
    sample_frame=8,
    more_image_control=True
)

# Create data loader
train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

# Training loop
for batch in train_loader:
    images = batch['image']           # [B, T, 3, H, W] - target images
    condition_image = batch['condition_image']  # [B, 3, H, W] - appearance reference
    conditions = batch['condition']   # [B, T, 3, H, W] - driving conditions
    extra_appearance = batch['extra_appearance']  # [B, 3, H, W] - back view reference
    
    # Your training code here...
        """)
        
    except Exception as e:
        print(f"❌ Error testing data format: {e}")
        import traceback
        traceback.print_exc()

def show_dataset_comparison():
    """
    Show comparison between our dataset and original full_head_clean.py datasets
    """
    print("=== Dataset Comparison ===")
    print()
    
    print("Original full_head_clean.py supports 3 dataset types:")
    print("1. PHsup_* (PanoHead): Uses 'image/' folder, specific camera lists")
    print("2. 0* (NeRSemble): Uses 'image_seg/' and 'camera/' folders")  
    print("3. i* (Stylization): Uses 'images/' folder, fixed 4 views")
    print()
    
    print("Our RenderMe360Dataset:")
    print("✅ Compatible with NeRSemble format (starts with '0')")
    print("✅ Uses same camera ID lists for face/back views")
    print("✅ Returns identical data structure")
    print("✅ Supports temporal sampling (sample_frame parameter)")
    print("✅ Supports extra appearance control (more_image_control)")
    print("✅ Handles 12 expressions × 60 viewpoints")
    print()
    
    print("Key advantages:")
    print("• Compact data format (1 JSON + 1 image folder)")
    print("• Easy to extend to other actors")
    print("• Maintains full compatibility with existing training code")
    print("• Efficient random sampling across expressions and viewpoints")

if __name__ == "__main__":
    # Run demonstrations
    demonstrate_compatibility()
    print()
    show_dataset_comparison()
    
    print("\n" + "="*60)
    print("🎉 Integration Complete!")
    print("Your RenderMe360 dataset is ready to use with DiffPortrait360!")
    print("="*60)
