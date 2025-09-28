#!/usr/bin/env python3
"""
Example script for using the compact 0018 data
"""

import json
import cv2
import numpy as np
from pathlib import Path

def load_0018_compact_data(data_dir="./0018_data_compact"):
    """
    Load the compact 0018 data
    
    Args:
        data_dir (str): Path to the compact data directory
        
    Returns:
        dict: Master data dictionary
    """
    data_path = Path(data_dir)
    json_file = data_path / "0018_data.json"
    
    if not json_file.exists():
        raise FileNotFoundError(f"Data file not found: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data

def get_image(data, expression, camera_id, data_dir="./0018_data_compact"):
    """
    Get a specific image
    
    Args:
        data (dict): Master data dictionary
        expression (str): Expression ID (e.g., 'e0')
        camera_id (str): Camera ID (e.g., '00')
        data_dir (str): Path to the compact data directory
        
    Returns:
        np.ndarray: Image array
    """
    if expression not in data['expressions']:
        raise ValueError(f"Expression {expression} not found")
    
    if camera_id not in data['expressions'][expression]['cameras']:
        raise ValueError(f"Camera {camera_id} not found for expression {expression}")
    
    image_path = data['expressions'][expression]['cameras'][camera_id]['image_path']
    full_path = Path(data_dir) / image_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Image file not found: {full_path}")
    
    image = cv2.imread(str(full_path))
    return image

def get_camera_calibration(data, camera_id):
    """
    Get camera calibration parameters
    
    Args:
        data (dict): Master data dictionary
        camera_id (str): Camera ID (e.g., '00')
        
    Returns:
        dict: Calibration parameters (K, D, RT)
    """
    if camera_id not in data['calibrations']:
        raise ValueError(f"Camera {camera_id} calibration not found")
    
    return data['calibrations'][camera_id]

def demo_usage():
    """
    Demonstrate how to use the compact data
    """
    print("=== RenderMe360 Actor 0018 Compact Data Demo ===")
    
    # Load data
    data = load_0018_compact_data()
    
    # Print basic info
    print(f"Actor ID: {data['actor_id']}")
    print(f"Actor Info: {data['actor_info']}")
    print(f"Summary: {data['summary']}")
    print()
    
    # List all expressions
    expressions = list(data['expressions'].keys())
    print(f"Available expressions: {expressions}")
    print()
    
    # Get a specific image
    expression = 'e0'
    camera_id = '25'
    
    print(f"Loading image: {expression}, camera {camera_id}")
    image = get_image(data, expression, camera_id)
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print()
    
    # Get camera calibration
    print(f"Camera {camera_id} calibration:")
    calib = get_camera_calibration(data, camera_id)
    print(f"  K (intrinsic): {np.array(calib['K']).shape}")
    print(f"  D (distortion): {np.array(calib['D']).shape}")
    print(f"  RT (extrinsic): {np.array(calib['RT']).shape}")
    print()
    
    # Show statistics for each expression
    print("Expression statistics:")
    for exp in expressions:
        exp_data = data['expressions'][exp]
        num_cameras = len(exp_data['cameras'])
        num_frames = exp_data['num_frames']
        print(f"  {exp}: {num_cameras} cameras, {num_frames} frames")
    
    print("\n=== Demo completed! ===")

def batch_process_example():
    """
    Example of batch processing all images
    """
    print("=== Batch Processing Example ===")
    
    data = load_0018_compact_data()
    
    # Process all expressions and cameras
    for expression in data['expressions']:
        print(f"Processing expression {expression}...")
        
        exp_data = data['expressions'][expression]
        cameras = exp_data['cameras']
        
        for camera_id in cameras:
            # Get image info without loading the actual image
            image_info = cameras[camera_id]
            image_filename = image_info['image_filename']
            image_shape = image_info['image_shape']
            
            print(f"  Camera {camera_id}: {image_filename}, shape: {image_shape}")
            
            # You can load and process the image here if needed:
            # image = get_image(data, expression, camera_id)
            # # Process image...
    
    print("Batch processing example completed!")

if __name__ == "__main__":
    # Run demo
    demo_usage()
    print()
    
    # Run batch processing example
    batch_process_example()
