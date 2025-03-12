"""
Auto training script for OpenPCDet models.
Sequentially trains multiple CenterPoint configurations on KITTI dataset.
"""

import os
import sys
import argparse
import subprocess
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model configurations to train
MODEL_CONFIGS = [
    'cfgs/kitti_models/Centerpoint_PP.yaml',
    'cfgs/kitti_models/CenterPoint_DynPP_SE.yaml', 
    'cfgs/kitti_models/Centerpoint_Voxel.yaml',
    'cfgs/kitti_models/Centerpoint_Voxel_SE.yaml',
    # 'cfgs/kitti_models/CenterpointRCNN_DynPP_SE.yaml',
]

def run_training(tools_path, config_file):
    """Run training with specified configuration"""
    start_time = time.time()
    model_name = os.path.basename(config_file).split('.')[0]
    
    logger.info(f"{'=' * 30}")
    logger.info(f"Starting training: {model_name}")
    logger.info(f"Config file: {config_file}")
    logger.info(f"{'=' * 30}")
    
    # Use relative paths for execution in tools directory
    train_script = 'train.py'
    
    # Save current directory
    original_dir = os.getcwd()
    
    try:
        # Change to tools directory to resolve relative paths in configs
        os.chdir(tools_path)
        
        # Build command with relative paths
        cmd = ['python', train_script, '--cfg_file', config_file]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output real-time
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            duration = time.time() - start_time
            logger.info(f"Training completed successfully for {model_name}")
            logger.info(f"Duration: {duration/60:.2f} minutes")
            return True
        else:
            logger.error(f"Training failed for {model_name} with return code {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error running training for {model_name}: {e}")
        return False
    finally:
        # Always return to original directory
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(description='Run training for OpenPCDet models')
    parser.add_argument('--tools_path', default='./tools', help='Path to the tools directory')
    parser.add_argument('--models', nargs='+', choices=range(len(MODEL_CONFIGS)), type=int,
                        help=f'Indices of models to train (0-{len(MODEL_CONFIGS)-1}), default is all')
    parser.add_argument('--continue_training', action='store_true', 
                        help='Continue training next model even if one fails')
    args = parser.parse_args()
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.tools_path.startswith('./'):
        tools_path = os.path.abspath(os.path.join(script_dir, args.tools_path))
    else:
        tools_path = os.path.abspath(args.tools_path)
    
    logger.info(f"Using tools directory: {tools_path}")
    
    # Print available models and their indices
    logger.info("Available model configurations:")
    for i, cfg in enumerate(MODEL_CONFIGS):
        logger.info(f"  [{i}] {os.path.basename(cfg)}")
    
    # Select models to train
    selected_models = args.models if args.models is not None else range(len(MODEL_CONFIGS))
    logger.info(f"Selected models for training: {[os.path.basename(MODEL_CONFIGS[i]) for i in selected_models]}")
    
    # Run training for each selected model
    results = {}
    for idx in selected_models:
        if idx < 0 or idx >= len(MODEL_CONFIGS):
            logger.warning(f"Invalid model index: {idx}, skipping")
            continue
            
        config_file = MODEL_CONFIGS[idx]
        result = run_training(tools_path=tools_path, config_file=config_file)
        
        model_name = os.path.basename(config_file).split('.')[0]
        results[model_name] = result
        
        if not result and not args.continue_training:
            logger.error(f"Training failed for {model_name}, stopping as requested")
            break
    
    # Print summary of results
    logger.info("\n" + "=" * 40)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 40)
    
    success_count = sum(1 for result in results.values() if result)
    logger.info(f"Models trained successfully: {success_count}/{len(results)}")
    
    for model, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{model}: {status}")
    
    logger.info("=" * 40)

if __name__ == "__main__":
    main()