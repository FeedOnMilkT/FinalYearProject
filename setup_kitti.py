"""
Automatic download and setup script for KITTI dataset in OpenPCDet.

This script:
1. Downloads the KITTI 3D object detection dataset
2. Extracts the data to the correct structure
3. Creates the necessary directory structure and dataset splits
4. Sets up the dataset for use with OpenPCDet
"""

import os
import sys
import argparse
import shutil
import urllib.request
import zipfile
import logging
from pathlib import Path
import numpy as np
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# KITTI dataset URLs
KITTI_URLS = [
    ('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip', 'data_object_velodyne.zip'),
    ('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip', 'data_object_calib.zip'),
    ('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip', 'data_object_label_2.zip'),
    ('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip', 'data_object_image_2.zip')
]

# Output path for the dataset (relative to the project root)
DEFAULT_OUTPUT_PATH = './data/kitti'

class DownloadProgressBar:
    """Progress bar for downloads"""
    def __init__(self, url):
        self.url = url
        self.filename = os.path.basename(url)
        self.downloaded_bytes = 0
        self.total_size = 0
        self.start_time = time.time()
        self.last_print_time = 0
    
    def __call__(self, count, block_size, total_size):
        self.total_size = total_size
        self.downloaded_bytes = count * block_size
        
        # Update progress every 0.5 seconds
        current_time = time.time()
        if current_time - self.last_print_time > 0.5:
            self.last_print_time = current_time
            
            # Calculate percentage and speed
            percent = min(100, self.downloaded_bytes * 100 // self.total_size) if self.total_size > 0 else 0
            elapsed_time = current_time - self.start_time
            speed = self.downloaded_bytes / (1024 * 1024 * elapsed_time) if elapsed_time > 0 else 0
            
            # Print progress
            sys.stdout.write(f"\r{self.filename}: {percent}% | {self.downloaded_bytes/(1024*1024):.1f}MB of {self.total_size/(1024*1024):.1f}MB | {speed:.1f} MB/s")
            sys.stdout.flush()

def download_file(url, output_path, force_download=False):
    """Download a file from URL to the specified output path."""
    if os.path.exists(output_path) and not force_download:
        logger.info(f"File already exists at {output_path}, skipping download")
        return
    
    logger.info(f"Downloading {url} to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        progress_bar = DownloadProgressBar(url)
        urllib.request.urlretrieve(url, output_path, reporthook=progress_bar)
        print()  # New line after progress bar
        logger.info(f"Successfully downloaded {url}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def extract_file(input_path, output_dir, delete_after=True):
    """Extract a compressed file to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Extracting {input_path} to {output_dir}")
    
    try:
        if input_path.endswith('.zip'):
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                # Get total number of files for progress tracking
                total_files = len(zip_ref.namelist())
                extracted_files = 0
                
                for file in zip_ref.namelist():
                    zip_ref.extract(file, output_dir)
                    extracted_files += 1
                    if extracted_files % 100 == 0 or extracted_files == total_files:
                        percent = min(100, extracted_files * 100 // total_files)
                        sys.stdout.write(f"\rExtracting {os.path.basename(input_path)}: {percent}%")
                        sys.stdout.flush()
                
                print()  # New line after progress bar
        else:
            logger.error(f"Unsupported file format: {input_path}")
            return
        
        logger.info(f"Successfully extracted {input_path}")
        
        # Delete zip file after extraction if requested
        if delete_after:
            logger.info(f"Deleting archive file {input_path}")
            os.remove(input_path)
            
    except Exception as e:
        logger.error(f"Failed to extract {input_path}: {e}")
        raise

def create_kitti_structure(kitti_root):
    """Create necessary directory structure for KITTI dataset."""
    logger.info("Creating KITTI directory structure")
    
    # Create required directories
    directories = [
        os.path.join(kitti_root, 'training', 'velodyne'),
        os.path.join(kitti_root, 'training', 'calib'),
        os.path.join(kitti_root, 'training', 'label_2'),
        os.path.join(kitti_root, 'training', 'image_2'),
        os.path.join(kitti_root, 'testing', 'velodyne'),
        os.path.join(kitti_root, 'testing', 'calib'),
        os.path.join(kitti_root, 'testing', 'image_2'),
        os.path.join(kitti_root, 'ImageSets')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Created KITTI directory structure")

def organize_kitti_files(kitti_root, temp_dir):
    """Organize extracted KITTI files into the proper structure."""
    logger.info("Organizing KITTI files")
    
    # Define source and destination folders
    train_dirs = ['velodyne', 'calib', 'label_2', 'image_2']
    test_dirs = ['velodyne', 'calib', 'image_2']
    
    # Process training data
    for folder in train_dirs:
        src_folder = os.path.join(temp_dir, 'training', folder)
        dst_folder = os.path.join(kitti_root, 'training', folder)
        
        if os.path.exists(src_folder):
            logger.info(f"Moving {folder} training files")
            os.makedirs(dst_folder, exist_ok=True)
            
            file_count = len([f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))])
            processed = 0
            
            for filename in os.listdir(src_folder):
                src_file = os.path.join(src_folder, filename)
                dst_file = os.path.join(dst_folder, filename)
                
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    processed += 1
                    
                    # Print progress
                    if processed % 100 == 0 or processed == file_count:
                        percent = min(100, processed * 100 // file_count) if file_count > 0 else 100
                        sys.stdout.write(f"\rMoving {folder} training files: {percent}%")
                        sys.stdout.flush()
            
            print()  # New line after progress
    
    # Process testing data
    for folder in test_dirs:
        src_folder = os.path.join(temp_dir, 'testing', folder)
        dst_folder = os.path.join(kitti_root, 'testing', folder)
        
        if os.path.exists(src_folder):
            logger.info(f"Moving {folder} testing files")
            os.makedirs(dst_folder, exist_ok=True)
            
            file_count = len([f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))])
            processed = 0
            
            for filename in os.listdir(src_folder):
                src_file = os.path.join(src_folder, filename)
                dst_file = os.path.join(dst_folder, filename)
                
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    processed += 1
                    
                    # Print progress
                    if processed % 100 == 0 or processed == file_count:
                        percent = min(100, processed * 100 // file_count) if file_count > 0 else 100
                        sys.stdout.write(f"\rMoving {folder} testing files: {percent}%")
                        sys.stdout.flush()
            
            print()  # New line after progress
    
    logger.info("Successfully organized KITTI files")

def create_imagesets(kitti_root):
    """Create train/val/test splits for KITTI if they don't already exist."""
    logger.info("Checking dataset splits")
    
    # Create ImageSets directory
    imagesets_dir = os.path.join(kitti_root, 'ImageSets')
    os.makedirs(imagesets_dir, exist_ok=True)
    
    # Check if split files already exist
    if all(os.path.exists(os.path.join(imagesets_dir, f"{split}.txt")) for split in ['train', 'val', 'test']):
        logger.info("Dataset splits already exist, preserving existing files")
        return
    
    # Get training sample IDs
    train_velodyne_dir = os.path.join(kitti_root, 'training', 'velodyne')
    if not os.path.exists(train_velodyne_dir):
        logger.error(f"Training velodyne directory {train_velodyne_dir} doesn't exist")
        return
    
    # Get sample IDs (remove file extension)
    train_samples = [os.path.splitext(f)[0] for f in os.listdir(train_velodyne_dir) if f.endswith('.bin')]
    train_samples.sort()
    
    # Get testing sample IDs
    test_velodyne_dir = os.path.join(kitti_root, 'testing', 'velodyne')
    test_samples = []
    if os.path.exists(test_velodyne_dir):
        test_samples = [os.path.splitext(f)[0] for f in os.listdir(test_velodyne_dir) if f.endswith('.bin')]
        test_samples.sort()
    
    # Split training into train and val (80/20 split)
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(train_samples))
    split_idx = int(len(train_samples) * 0.8)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    train_split = [train_samples[i] for i in train_idx]
    val_split = [train_samples[i] for i in val_idx]
    
    # Sort for better readability
    train_split.sort()
    val_split.sort()
    
    # Check individual split files and only create missing ones
    if not os.path.exists(os.path.join(imagesets_dir, 'train.txt')):
        with open(os.path.join(imagesets_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(train_split))
        logger.info(f"Created train split with {len(train_split)} samples")
    else:
        logger.info("Using existing train.txt file")
    
    if not os.path.exists(os.path.join(imagesets_dir, 'val.txt')):
        with open(os.path.join(imagesets_dir, 'val.txt'), 'w') as f:
            f.write('\n'.join(val_split))
        logger.info(f"Created val split with {len(val_split)} samples")
    else:
        logger.info("Using existing val.txt file")
    
    if not os.path.exists(os.path.join(imagesets_dir, 'test.txt')):
        with open(os.path.join(imagesets_dir, 'test.txt'), 'w') as f:
            f.write('\n'.join(test_samples))
        logger.info(f"Created test split with {len(test_samples)} samples")
    else:
        logger.info("Using existing test.txt file")

def main():
    parser = argparse.ArgumentParser(description='Download and setup KITTI dataset for OpenPCDet')
    parser.add_argument('--output_path', default=DEFAULT_OUTPUT_PATH, help='Path to save the dataset')
    parser.add_argument('--temp_dir', default='./data/temp/kitti', help='Temporary directory for downloads')
    parser.add_argument('--force_download', action='store_true', help='Force re-download of files')
    parser.add_argument('--skip_download', action='store_true', help='Skip downloading files')
    parser.add_argument('--keep_temp', action='store_true', help='Keep temporary files after processing')
    parser.add_argument('--preserve_splits', action='store_true', help='Preserve existing dataset splits')
    parser.add_argument('--keep_archives', action='store_true', help='Keep zip files after extraction')
    args = parser.parse_args()
    
    # Get script directory for reference
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle paths - make them absolute and ensure they exist
    if args.output_path.startswith('./'):
        output_path = os.path.abspath(os.path.join(script_dir, args.output_path))
    else:
        output_path = os.path.abspath(args.output_path)
    
    if args.temp_dir.startswith('./'):
        temp_dir = os.path.abspath(os.path.join(script_dir, args.temp_dir))
    else:
        temp_dir = os.path.abspath(args.temp_dir)
    
    # Ensure directories exist
    os.makedirs(temp_dir, exist_ok=True)
    
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Setting up KITTI dataset at {output_path}")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Step 1: Download dataset
        if not args.skip_download:
            os.makedirs(temp_dir, exist_ok=True)
            for url, filename in KITTI_URLS:
                download_path = os.path.join(temp_dir, filename)
                download_file(url, download_path, args.force_download)
                extract_file(download_path, temp_dir, not args.keep_archives)
        
        # Step 2: Create directory structure
        create_kitti_structure(output_path)
        
        # Step 3: Organize files
        organize_kitti_files(output_path, temp_dir)
        
        # Step 4: Create dataset splits
        create_imagesets(output_path)
        
        # Clean up temporary files if not keeping them
        if not args.keep_temp and os.path.exists(temp_dir) and not args.skip_download:
            logger.info(f"Cleaning up temporary directory {temp_dir}")
            shutil.rmtree(temp_dir)
        
        logger.info("===============================")
        logger.info("KITTI dataset setup completed!")
        logger.info(f"Dataset location: {output_path}")
        logger.info("===============================")
        logger.info("\nTo use this dataset with OpenPCDet:")
        logger.info("1. Ensure the pcdet package is installed")
        logger.info("2. Use the KITTI dataset configuration in your training/testing scripts")
        logger.info("3. Sample command: python tools/train.py --cfg_file tools/cfgs/kitti_models/second.yaml")
    
    except Exception as e:
        logger.error(f"Error setting up KITTI dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()