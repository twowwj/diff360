import json
import cv2
import numpy as np
import os
import random
try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms as T
except ImportError:
    print("PyTorch not available, using mock classes for testing")
    class Dataset:
        pass
    class T:
        @staticmethod
        def Compose(transforms):
            return lambda x: x
        @staticmethod
        def ToTensor():
            return lambda x: x
        @staticmethod
        def Normalize(mean, std):
            return lambda x: x
    torch = None

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RenderMe360Dataset(Dataset):
    """
    Dataset for RenderMe360 compact data format
    Compatible with DiffPortrait360's full_head_clean.py interface
    """
    
    def __init__(self, data_dir, image_transform=None, sample_frame=8, more_image_control=True):
        """
        Args:
            data_dir (str): Path to compact data directory (e.g., './0018_data_compact')
            image_transform: Image transformation pipeline
            sample_frame (int): Number of frames to sample for temporal training
            more_image_control (bool): Whether to include extra appearance control
        """
        self.data_dir = data_dir
        self.transform = image_transform
        self.sample_frame = sample_frame
        self.more_image_control = more_image_control
        
        # Load master data
        json_path = os.path.join(data_dir, "0018_data.json")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Extract expressions and camera info
        self.expressions = list(self.data['expressions'].keys())
        self.actor_id = self.data['actor_id']
        
        # Define camera lists compatible with NeRSemble dataset (startswith '0')
        # These match the potential_NS_fine_face_list and potential_NS_fine_back_list
        self.face_cameras = ['16', '18', '19', '25', '26', '28', '31', '55', '56']
        self.back_cameras = ['59', '50', '49', '48', '46', '45', '01', '00', '02']
        
        # All available cameras (0-59)
        self.all_cameras = [f"{i:02d}" for i in range(60)]
        
        print(f"Loaded RenderMe360 dataset:")
        print(f"  Actor ID: {self.actor_id}")
        print(f"  Expressions: {len(self.expressions)} ({', '.join(self.expressions)})")
        print(f"  Face cameras: {len(self.face_cameras)} cameras")
        print(f"  Back cameras: {len(self.back_cameras)} cameras")
        print(f"  Total cameras: {len(self.all_cameras)} cameras")
    
    def __len__(self):
        # Return a large number for infinite sampling during training
        return 100000
    
    def __getitem__(self, idx):
        """
        Returns data compatible with DiffPortrait360's expected format:
        {
            'image': targets,           # [sample_frame, 3, H, W] - target images
            'condition_image': condition_image,  # [3, H, W] - appearance reference
            'condition': conditions,    # [sample_frame, 3, H, W] - driving conditions
            'text_bg': prompt,         # text prompt (empty)
            'text_blip': prompt,       # text prompt (empty)
            'extra_appearance': more_appearance_frame  # [3, H, W] - back view reference
        }
        """
        if self.sample_frame == 1:
            raise KeyError("sample_frame should be at least larger than 1")
        
        # Randomly select an expression
        expression = random.choice(self.expressions)
        exp_data = self.data['expressions'][expression]
        
        # Get available cameras for this expression
        available_cameras = list(exp_data['cameras'].keys())
        
        # Filter cameras based on our predefined lists
        valid_face_cameras = [cam for cam in self.face_cameras if cam in available_cameras]
        valid_back_cameras = [cam for cam in self.back_cameras if cam in available_cameras]
        valid_all_cameras = [cam for cam in self.all_cameras if cam in available_cameras]
        
        # Ensure we have enough cameras
        if len(valid_face_cameras) == 0:
            valid_face_cameras = valid_all_cameras[:len(self.face_cameras)]
        if len(valid_back_cameras) == 0:
            valid_back_cameras = valid_all_cameras[-len(self.back_cameras):]
        
        # Shuffle for randomness
        random.shuffle(valid_all_cameras)
        random.shuffle(valid_face_cameras)
        random.shuffle(valid_back_cameras)
        
        # Select appearance frame (from face cameras)
        appearance_camera = random.choice(valid_face_cameras)
        appearance_image_info = exp_data['cameras'][appearance_camera]
        appearance_image_path = os.path.join(self.data_dir, appearance_image_info['image_path'])
        appearance_image = self.read_image(appearance_image_path)
        
        # Select frames for temporal sequence
        # Use pick_frames logic similar to original dataset
        interval = random.randint(2, 6)
        frame_cameras = self.pick_cameras(
            num_cameras=self.sample_frame, 
            available_cameras=valid_all_cameras, 
            interval=interval
        )
        
        print(f'Expression: {expression}, Cameras: {frame_cameras[:4]}, Interval: {interval}')
        
        # Load target images and conditions
        targets = []
        conditions = []
        
        for camera_id in frame_cameras:
            # Target image (same as condition for our dataset)
            camera_info = exp_data['cameras'][camera_id]
            image_path = os.path.join(self.data_dir, camera_info['image_path'])
            target_image = self.read_image(image_path)
            
            # Condition image (same as target for our dataset)
            condition_image = target_image.copy()
            
            # Apply transforms
            if self.transform is not None:
                target_image = self.transform(target_image)
                condition_image = self.transform(condition_image)
            
            targets.append(target_image)
            conditions.append(condition_image)
        
        # Stack tensors (if torch is available)
        if torch is not None:
            targets = torch.stack(targets)
            conditions = torch.stack(conditions)
        else:
            targets = np.stack(targets)
            conditions = np.stack(conditions)
        
        # Apply transform to appearance image
        if self.transform is not None:
            condition_image = self.transform(appearance_image)
        
        # Prepare result
        prompt = ''
        res = {
            'image': targets,
            'condition_image': condition_image,
            'condition': conditions,
            'text_bg': prompt,
            'text_blip': prompt
        }
        
        # Add extra appearance control (back view)
        if self.more_image_control:
            back_camera = random.choice(valid_back_cameras)
            back_image_info = exp_data['cameras'][back_camera]
            back_image_path = os.path.join(self.data_dir, back_image_info['image_path'])
            back_image = self.read_image(back_image_path)
            
            if self.transform is not None:
                back_image = self.transform(back_image)
            
            res['extra_appearance'] = back_image
        
        return res
    
    def read_image(self, path):
        """Read and preprocess image"""
        # Open the image file
        image = Image.open(path)
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to 512x512 (matching original dataset)
        image = image.resize((512, 512))
        
        # Convert the PIL image to a NumPy array
        image_np = np.array(image)
        
        return image_np
    
    def pick_cameras(self, num_cameras, available_cameras, interval=None):
        """
        Pick cameras with interval sampling (similar to pick_frames in original)
        """
        if interval is None:
            interval = 1
        
        total_cameras = len(available_cameras)
        
        # Random starting camera
        start_idx = random.randint(0, total_cameras - 1)
        
        # Select cameras with interval
        selected_cameras = []
        for i in range(num_cameras):
            camera_idx = (start_idx + i * interval) % total_cameras
            selected_cameras.append(available_cameras[camera_idx])
        
        return selected_cameras

class RenderMe360SingleFrameDataset(Dataset):
    """
    Single frame version for inference/testing
    """
    
    def __init__(self, data_dir, image_transform=None, more_image_control=False):
        self.data_dir = data_dir
        self.transform = image_transform
        self.more_image_control = more_image_control
        
        # Load master data
        json_path = os.path.join(data_dir, "0018_data.json")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.expressions = list(self.data['expressions'].keys())
        self.actor_id = self.data['actor_id']
        
        # Camera lists
        self.face_cameras = ['16', '18', '19', '25', '26', '28', '31', '55', '56']
        self.back_cameras = ['59', '50', '49', '48', '46', '45', '01', '00', '02']
        self.all_cameras = [f"{i:02d}" for i in range(60)]
    
    def __len__(self):
        return 100000
    
    def __getitem__(self, idx):
        # Similar to temporal version but returns single frame
        expression = random.choice(self.expressions)
        exp_data = self.data['expressions'][expression]
        
        available_cameras = list(exp_data['cameras'].keys())
        valid_face_cameras = [cam for cam in self.face_cameras if cam in available_cameras]
        valid_back_cameras = [cam for cam in self.back_cameras if cam in available_cameras]
        
        if len(valid_face_cameras) == 0:
            valid_face_cameras = available_cameras[:len(self.face_cameras)]
        if len(valid_back_cameras) == 0:
            valid_back_cameras = available_cameras[-len(self.back_cameras):]
        
        # Select appearance and target cameras
        appearance_camera = random.choice(valid_face_cameras)
        target_camera = random.choice(valid_back_cameras)
        
        # Load images
        appearance_info = exp_data['cameras'][appearance_camera]
        target_info = exp_data['cameras'][target_camera]
        
        appearance_path = os.path.join(self.data_dir, appearance_info['image_path'])
        target_path = os.path.join(self.data_dir, target_info['image_path'])
        
        appearance_image = self.read_image(appearance_path)
        target_image = self.read_image(target_path)
        condition_image = target_image.copy()  # Same as target for our dataset
        
        # Apply transforms
        prompt = ''
        if self.transform is not None:
            target = self.transform(target_image)
            condition_image = self.transform(appearance_image)
            condition = self.transform(condition_image)
        
        res = {
            'image': target,
            'condition_image': condition_image,
            'condition': condition,
            'text_bg': prompt,
            'text_blip': prompt
        }
        
        return res
    
    def read_image(self, path):
        """Read and preprocess image"""
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((512, 512))
        image_np = np.array(image)
        return image_np


if __name__ == "__main__":
    # Test the dataset without transforms first
    print("Testing dataset without transforms...")

    # Test temporal dataset
    dataset = RenderMe360Dataset(
        data_dir="./0018_data_compact",
        image_transform=None,  # No transforms for testing
        sample_frame=4,  # Smaller sample for testing
        more_image_control=True
    )

    print(f"Dataset length: {len(dataset)}")

    # Test single sample
    try:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape if hasattr(sample['image'], 'shape') else type(sample['image'])}")
        print(f"Condition image shape: {sample['condition_image'].shape if hasattr(sample['condition_image'], 'shape') else type(sample['condition_image'])}")
        print(f"Condition shape: {sample['condition'].shape if hasattr(sample['condition'], 'shape') else type(sample['condition'])}")
        if 'extra_appearance' in sample:
            print(f"Extra appearance shape: {sample['extra_appearance'].shape if hasattr(sample['extra_appearance'], 'shape') else type(sample['extra_appearance'])}")
        print("✅ Dataset test successful!")
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
