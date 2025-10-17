"""
GPU Verification Script for PyTorch
Checks CUDA availability and provides installation instructions if needed.
"""

import sys

try:
    import torch
    
    print("=" * 70)
    print("PyTorch GPU Verification")
    print("=" * 70)
    
    # PyTorch version
    print(f"\n✓ PyTorch Version: {torch.__version__}")
    
    # CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: YES")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        # GPU details
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
            print(f"    - Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    - Multi-Processors: {props.multi_processor_count}")
        
        # Test GPU
        print(f"\n✓ Testing GPU...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✓ GPU test successful! Result shape: {z.shape}")
        
        # Current device
        print(f"\n✓ Current CUDA device: {torch.cuda.current_device()}")
        print(f"✓ CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        print("\n" + "=" * 70)
        print("SUCCESS! Your system is ready for GPU training! 🚀")
        print("=" * 70)
        
    else:
        print(f"✗ CUDA Available: NO")
        print("\n" + "=" * 70)
        print("ERROR: CUDA is not available!")
        print("=" * 70)
        
        # Check if it's CPU-only build
        if '+cpu' in torch.__version__:
            print("\n⚠ You have a CPU-only version of PyTorch installed.")
            print("\nTo install CUDA-enabled PyTorch, run:")
            print("  1. Uninstall current PyTorch:")
            print("     pip uninstall torch torchvision torchaudio")
            print("\n  2. Install CUDA-enabled PyTorch (CUDA 12.1):")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("\n  3. Or for CUDA 11.8:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        else:
            print("\n⚠ PyTorch has CUDA support, but CUDA is not detected.")
            print("\nPossible reasons:")
            print("  1. NVIDIA GPU drivers not installed")
            print("  2. CUDA toolkit not installed")
            print("  3. GPU not detected by the system")
            print("\nPlease:")
            print("  1. Check if your GPU is detected: run 'nvidia-smi'")
            print("  2. Install/update NVIDIA drivers: https://www.nvidia.com/drivers")
        
        print("\n" + "=" * 70)
        sys.exit(1)

except ImportError:
    print("=" * 70)
    print("ERROR: PyTorch is not installed!")
    print("=" * 70)
    print("\nTo install PyTorch with CUDA support, run:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\n" + "=" * 70)
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Error during GPU verification: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
