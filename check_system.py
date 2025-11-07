#!/usr/bin/env python3
"""
System Requirements Checker
Quick check to see if your system can handle DOTS OCR locally
"""

import sys
import subprocess
import shutil

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    print("=" * 60)
    print("1. NVIDIA GPU Check")
    print("=" * 60)

    if not shutil.which('nvidia-smi'):
        print("✗ nvidia-smi not found")
        print("  No NVIDIA GPU detected or drivers not installed")
        return False

    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split('\n')
        for idx, line in enumerate(lines):
            parts = line.split(',')
            if len(parts) >= 2:
                gpu_name = parts[0].strip()
                memory_str = parts[1].strip()
                # Extract memory in MB
                memory_mb = int(''.join(filter(str.isdigit, memory_str)))
                memory_gb = memory_mb / 1024

                print(f"✓ GPU {idx}: {gpu_name}")
                print(f"  VRAM: {memory_gb:.2f} GB")

                if memory_gb < 12:
                    print(f"  ⚠ WARNING: Less than 12GB VRAM")
                    print(f"  May not be sufficient for DOTS OCR")
                    return False
                else:
                    print(f"  ✓ Sufficient VRAM for OCR")
                    return True

    except subprocess.CalledProcessError:
        print("✗ Error running nvidia-smi")
        return False

    return False

def check_cuda():
    """Check CUDA availability via Python"""
    print("\n" + "=" * 60)
    print("2. CUDA/PyTorch Check")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")

            gpu_count = torch.cuda.device_count()
            print(f"✓ Available GPUs: {gpu_count}")

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")

            return True
        else:
            print("✗ CUDA not available in PyTorch")
            return False

    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision")
        return False

def check_system_memory():
    """Check system RAM"""
    print("\n" + "=" * 60)
    print("3. System Memory Check")
    print("=" * 60)

    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_gb = mem.total / 1024**3

        print(f"Total RAM: {mem_gb:.2f} GB")

        if mem_gb < 16:
            print("⚠ WARNING: Less than 16GB RAM")
            print("  May experience issues with large batches")
            return False
        else:
            print("✓ Sufficient RAM")
            return True

    except ImportError:
        print("✗ psutil not installed (optional)")
        print("  Install with: pip install psutil")
        return None

def check_disk_space():
    """Check available disk space"""
    print("\n" + "=" * 60)
    print("4. Disk Space Check")
    print("=" * 60)

    try:
        import shutil

        total, used, free = shutil.disk_usage(".")
        free_gb = free / 1024**3

        print(f"Free disk space: {free_gb:.2f} GB")

        if free_gb < 50:
            print("⚠ WARNING: Less than 50GB free")
            print("  You'll need space for:")
            print("  - Model weights (~10GB)")
            print("  - Your data")
            print("  - OCR output")
            return False
        else:
            print("✓ Sufficient disk space")
            return True

    except Exception as e:
        print(f"✗ Could not check disk space: {e}")
        return None

def check_internet():
    """Check internet connectivity for model download"""
    print("\n" + "=" * 60)
    print("5. Internet Connectivity Check")
    print("=" * 60)

    try:
        import socket
        socket.create_connection(("huggingface.co", 443), timeout=5)
        print("✓ Can reach HuggingFace (model hosting)")
        return True
    except OSError:
        print("✗ Cannot reach HuggingFace")
        print("  You'll need internet to download the model")
        return False

def estimate_processing_time(gpu_name, num_pages=150000):
    """Estimate processing time based on GPU"""
    print("\n" + "=" * 60)
    print("6. Processing Time Estimate")
    print("=" * 60)

    # Speed estimates in pages per minute
    speed_map = {
        'RTX 4090': (15, 20),
        'RTX 4080': (12, 18),
        'RTX 3090': (10, 15),
        'RTX 3080': (8, 12),
        'RTX 3070': (6, 10),
        'RTX 3060': (5, 8),
        'A100': (30, 40),
        'A40': (15, 25),
        'A10': (10, 15),
        'T4': (5, 8),
    }

    speed = None
    for key in speed_map:
        if key in gpu_name:
            speed = speed_map[key]
            break

    if speed:
        min_speed, max_speed = speed
        min_hours = num_pages / max_speed / 60
        max_hours = num_pages / min_speed / 60

        print(f"Estimated speed: {min_speed}-{max_speed} pages/minute")
        print(f"Time for {num_pages:,} pages: {min_hours:.1f}-{max_hours:.1f} hours")
        print(f"That's approximately {min_hours/24:.1f}-{max_hours/24:.1f} days")

        if min_hours > 200:
            print("\n⚠ That's quite a long time!")
            print("  Consider cloud GPU for faster processing")
    else:
        print("Could not estimate speed for your GPU")
        print("Typical speeds: 5-40 pages/minute depending on GPU")

def main():
    print("\n" + "=" * 60)
    print("DOTS OCR - System Requirements Check")
    print("=" * 60)
    print()

    results = []

    # Run all checks
    has_gpu = check_nvidia_gpu()
    results.append(('GPU', has_gpu))

    has_cuda = check_cuda()
    results.append(('CUDA', has_cuda))

    has_ram = check_system_memory()
    results.append(('RAM', has_ram))

    has_disk = check_disk_space()
    results.append(('Disk', has_disk))

    has_inet = check_internet()
    results.append(('Internet', has_inet))

    # Get GPU name for time estimate
    if has_gpu:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  capture_output=True, text=True, check=True)
            gpu_name = result.stdout.strip().split('\n')[0]
            estimate_processing_time(gpu_name)
        except:
            pass

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)

    for name, result in results:
        if result is True:
            print(f"✓ {name}: PASS")
        elif result is False:
            print(f"✗ {name}: FAIL")
        else:
            print(f"? {name}: UNKNOWN")

    print()

    if passed >= 4 and has_gpu and has_cuda:
        print("=" * 60)
        print("✓ YOUR SYSTEM CAN RUN DOTS OCR LOCALLY")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Install dependencies:")
        print("   pip install -r requirements-ocr.txt")
        print()
        print("2. Run test:")
        print("   python test_ocr.py")
        print()
        print("3. Process images:")
        print("   python ocr_processor.py --data-dir ./data --output-dir ./ocr_output")

    elif has_gpu and not has_cuda:
        print("=" * 60)
        print("⚠ GPU FOUND BUT CUDA NOT CONFIGURED")
        print("=" * 60)
        print()
        print("You have a GPU but PyTorch can't use it.")
        print()
        print("Fix:")
        print("1. Install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print()
        print("2. Then run this check again")

    else:
        print("=" * 60)
        print("✗ CLOUD GPU RECOMMENDED")
        print("=" * 60)
        print()
        print("Your system doesn't meet the requirements for local processing.")
        print()
        print("Recommended cloud providers:")
        print("1. Vast.ai - $0.20-0.35/hour (~$33-108 total)")
        print("   https://vast.ai")
        print()
        print("2. RunPod - $0.50/hour (~$63-125 total)")
        print("   https://runpod.io")
        print()
        print("3. Lambda Labs - $1.10/hour (~$150-212 total)")
        print("   https://lambdalabs.com")
        print()
        print("See QUICKSTART_OCR.md for cloud setup instructions")

    print()

if __name__ == "__main__":
    main()
