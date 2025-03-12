import os
import subprocess
import sys

def run_command(cmd):
    print(f"Execute: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def setup_openpcdet():
    pcdet_dir = "/workspace/FinalYearProject"
    os.chdir(pcdet_dir)
    
    print("=== Install Dependencies ===")
    run_command("pip install numpy")
    run_command("pip install -r requirements.txt")

    print("=== Install Spconv for CUDA 12 ===")
    run_command("pip install spconv-cu120")
    
    # print("=== Insatll nuScenes devkit ===")
    # run_command("pip install nuscenes-devkit")
    
    print("=== Install torch-scatter ===")
    run_command("pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu121.html")
    
    print("=== Compile OpenPCDet ===")
    run_command("python setup.py develop")
    
    # print("=== Install SharedArray ===")
    # run_command("pip install SharedArray")
    

if __name__ == "__main__":
    setup_openpcdet()