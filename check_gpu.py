import torch
import subprocess
import os

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

print("=== PyTorch GPU Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

print("\n=== Environment Variables ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

print("\n=== System GPU Information ===")
print("nvidia-smi output:")
print(run_cmd("nvidia-smi"))

print("\n=== CUDA Installation ===")
print("Checking CUDA installation:")
print(run_cmd("which nvcc 2>/dev/null || echo 'nvcc not found'"))
print("\nChecking CUDA libraries:")
print(run_cmd("ldconfig -p | grep -i cuda || echo 'No CUDA libraries found'"))
