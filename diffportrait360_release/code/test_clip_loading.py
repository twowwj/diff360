#!/usr/bin/env python3
"""
Test script to verify CLIP model loading works correctly.
"""

import os
import sys
import torch

def test_clip_loading():
    """Test if CLIP model can be loaded successfully"""
    try:
        # Add the current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # Import the CLIP embedder
        from model_lib.ControlNet.ldm.modules.encoders.modules import FrozenCLIPEmbedder
        
        print("✓ Successfully imported FrozenCLIPEmbedder")
        
        # Test loading with default path
        print("Testing CLIP model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create the embedder
        embedder = FrozenCLIPEmbedder(device=device)
        print("✓ Successfully created FrozenCLIPEmbedder")
        
        # Test encoding some text
        test_text = ["a portrait of a person"]
        with torch.no_grad():
            encoded = embedder.encode(test_text)
            print(f"✓ Successfully encoded text. Output shape: {encoded.shape}")
        
        print("🎉 All tests passed! CLIP model loading works correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_clip_files():
    """Check if CLIP model files exist"""
    clip_path = "openai/clip-vit-large-patch14"
    
    print(f"Checking CLIP model files in: {clip_path}")
    
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]
    
    all_exist = True
    for file in required_files:
        file_path = os.path.join(clip_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file}: {size:,} bytes")
        else:
            print(f"✗ {file}: NOT FOUND")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("=== CLIP Model Loading Test ===")
    print()
    
    # Check if files exist
    print("1. Checking CLIP model files...")
    files_ok = check_clip_files()
    print()
    
    if not files_ok:
        print("❌ Some CLIP model files are missing!")
        sys.exit(1)
    
    # Test loading
    print("2. Testing CLIP model loading...")
    success = test_clip_loading()
    
    if success:
        print("\n🎉 CLIP model test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ CLIP model test failed!")
        sys.exit(1)
