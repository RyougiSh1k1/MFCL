#!/usr/bin/env python3
"""
Dataset setup script for MFCL project
Downloads and prepares datasets for federated continual learning experiments
"""

import os
import sys
import wget
import zipfile
import argparse
from pathlib import Path


def download_tiny_imagenet(data_dir="data"):
    """Download and extract Tiny ImageNet dataset"""
    print("Setting up Tiny ImageNet dataset...")
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    tiny_imagenet_path = data_path / "tiny-imagenet-200"
    zip_path = data_path / "tiny-imagenet-200.zip"
    
    # Check if already exists
    if tiny_imagenet_path.exists():
        print(f"Tiny ImageNet already exists at {tiny_imagenet_path}")
        return True
    
    # Download if zip doesn't exist
    if not zip_path.exists():
        print("Downloading Tiny ImageNet dataset...")
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        try:
            wget.download(url, str(zip_path))
            print(f"\nDownloaded to {zip_path}")
        except Exception as e:
            print(f"Error downloading Tiny ImageNet: {e}")
            print("Please download manually from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            return False
    
    # Extract
    print("Extracting Tiny ImageNet dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print(f"Extracted to {tiny_imagenet_path}")
        
        # Verify structure
        expected_dirs = ['train', 'val', 'test']
        for dir_name in expected_dirs:
            dir_path = tiny_imagenet_path / dir_name
            if dir_path.exists():
                print(f"✓ Found {dir_name} directory")
            else:
                print(f"✗ Missing {dir_name} directory")
        
        # Check train classes
        train_path = tiny_imagenet_path / "train"
        if train_path.exists():
            class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
            print(f"✓ Found {len(class_dirs)} class directories in train")
        
        # Check val annotations
        val_annotations = tiny_imagenet_path / "val" / "val_annotations.txt"
        if val_annotations.exists():
            print("✓ Found val_annotations.txt")
        else:
            print("✗ Missing val_annotations.txt")
        
        return True
        
    except Exception as e:
        print(f"Error extracting Tiny ImageNet: {e}")
        return False


def setup_cifar100():
    """CIFAR-100 setup info"""
    print("CIFAR-100 Setup:")
    print("CIFAR-100 will be automatically downloaded by torchvision when first used.")
    print("No manual setup required.")
    return True


def setup_super_imagenet():
    """SuperImageNet setup info"""
    print("SuperImageNet Setup:")
    print("SuperImageNet requires the full ImageNet dataset.")
    print("Please:")
    print("1. Download ImageNet from https://www.image-net.org/")
    print("2. Extract to your desired location")
    print("3. Ensure the structure is:")
    print("   <path>/")
    print("   ├── train/")
    print("   │   ├── n01440764/")
    print("   │   ├── n01443537/")
    print("   │   └── ...")
    print("   └── val/")
    print("       ├── n01440764/")
    print("       ├── n01443537/")
    print("       └── ...")
    print("4. Use --path <your_imagenet_path> when running experiments")
    return True


def verify_dataset(dataset_name, data_path="data"):
    """Verify dataset setup"""
    print(f"\nVerifying {dataset_name} dataset...")
    
    if dataset_name == "tiny_imagenet":
        tiny_path = Path(data_path) / "tiny-imagenet-200"
        if not tiny_path.exists():
            print(f"✗ Tiny ImageNet not found at {tiny_path}")
            return False
        
        # Check required directories
        required_dirs = ["train", "val"]
        for dir_name in required_dirs:
            dir_path = tiny_path / dir_name
            if not dir_path.exists():
                print(f"✗ Missing {dir_name} directory")
                return False
            print(f"✓ Found {dir_name} directory")
        
        # Check train classes
        train_path = tiny_path / "train"
        class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
        if len(class_dirs) == 0:
            print("✗ No class directories found in train")
            return False
        print(f"✓ Found {len(class_dirs)} class directories")
        
        # Check val annotations
        val_annotations = tiny_path / "val" / "val_annotations.txt"
        if not val_annotations.exists():
            print("✗ Missing val_annotations.txt")
            return False
        print("✓ Found val_annotations.txt")
        
        print(f"✓ Tiny ImageNet setup verified!")
        return True
        
    elif dataset_name == "cifar100":
        print("✓ CIFAR-100 will be verified automatically when used")
        return True
        
    elif dataset_name == "super_imagenet":
        print("Please verify SuperImageNet manually by checking your ImageNet path")
        return True
    
    else:
        print(f"Unknown dataset: {dataset_name}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup datasets for MFCL experiments")
    parser.add_argument("--dataset", type=str, choices=["tiny_imagenet", "cifar100", "super_imagenet", "all"],
                       default="all", help="Dataset to setup")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Directory to store datasets")
    parser.add_argument("--verify", action="store_true", 
                       help="Only verify existing datasets")
    
    args = parser.parse_args()
    
    print("MFCL Dataset Setup")
    print("="*50)
    
    success = True
    
    if args.verify:
        # Verification mode
        if args.dataset == "all":
            datasets = ["tiny_imagenet", "cifar100", "super_imagenet"]
        else:
            datasets = [args.dataset]
        
        for dataset in datasets:
            if not verify_dataset(dataset, args.data_dir):
                success = False
    else:
        # Setup mode
        if args.dataset in ["tiny_imagenet", "all"]:
            if not download_tiny_imagenet(args.data_dir):
                success = False
        
        if args.dataset in ["cifar100", "all"]:
            setup_cifar100()
        
        if args.dataset in ["super_imagenet", "all"]:
            setup_super_imagenet()
    
    print("\n" + "="*50)
    if success:
        print("Dataset setup completed successfully!")
        print(f"\nYou can now run experiments with:")
        print(f"python main.py --dataset=tiny_imagenet --path={args.data_dir}")
    else:
        print("Dataset setup encountered issues. Please check the output above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())