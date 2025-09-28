#!/usr/bin/env python3
"""
Data Summary Script for 0018 Generated Data
"""

import os
import json
from pathlib import Path

def summarize_0018_data(data_dir="./0018_generated_data"):
    """
    Summarize the generated data for actor 0018
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory {data_dir} does not exist!")
        return
    
    print("=" * 60)
    print("RenderMe360 Actor 0018 Data Summary")
    print("=" * 60)
    
    # Get all expression directories
    expressions = sorted([d.name for d in data_path.iterdir() if d.is_dir() and d.name.startswith('e')])
    
    print(f"Total Expressions: {len(expressions)}")
    print(f"Expressions: {', '.join(expressions)}")
    print()
    
    total_cameras = 0
    total_images = 0
    
    # Analyze each expression
    for exp in expressions:
        exp_path = data_path / exp
        print(f"Expression {exp}:")
        
        # Read basic info
        info_file = exp_path / "info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
            
            actor_info = info.get('actor_info', {})
            camera_info = info.get('camera_info', {})
            
            print(f"  Actor: {actor_info.get('gender', 'N/A')}, {actor_info.get('age', 'N/A')} years old")
            print(f"  Frames: {camera_info.get('num_frame', 'N/A')}")
            print(f"  Resolution: {camera_info.get('resolution', 'N/A')}")
        
        # Count cameras and images
        images_path = exp_path / "images"
        if images_path.exists():
            cameras = [d for d in images_path.iterdir() if d.is_dir() and d.name.startswith('cam_')]
            num_cameras = len(cameras)
            total_cameras += num_cameras
            
            # Count images
            num_images = 0
            for cam_dir in cameras:
                images = list(cam_dir.glob("*.jpg"))
                num_images += len(images)
            
            total_images += num_images
            
            print(f"  Cameras: {num_cameras}")
            print(f"  Images: {num_images}")
        
        # Check for additional data
        additional_data = []
        if (exp_path / "calibrations.json").exists():
            additional_data.append("Camera Calibrations")
        if (exp_path / "flame").exists():
            additional_data.append("FLAME Parameters")
        if (exp_path / "keypoints").exists():
            additional_data.append("3D Keypoints")
        
        if additional_data:
            print(f"  Additional Data: {', '.join(additional_data)}")
        
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Expressions: {len(expressions)}")
    print(f"Total Camera Views: {total_cameras} ({total_cameras // len(expressions)} per expression)")
    print(f"Total Images: {total_images}")
    print(f"Data Structure: {len(expressions)} expressions × 60 viewpoints = {len(expressions) * 60} camera positions")
    print()
    
    # Calculate data size
    try:
        import subprocess
        result = subprocess.run(['du', '-sh', data_dir], capture_output=True, text=True)
        if result.returncode == 0:
            size = result.stdout.strip().split()[0]
            print(f"Total Data Size: {size}")
    except:
        print("Could not calculate data size")
    
    print()
    print("Data successfully generated for RenderMe360 Actor 0018!")
    print("12 expressions × 60 viewpoints = 720 camera positions with sample images")

if __name__ == "__main__":
    summarize_0018_data()
