#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detect the type (CPU/GPU) of the PyTorch build installed in the virtual environment
"""

import sys

def detect_torch_type():
    """Detect the PyTorch type of the current environment"""
    try:
        import torch
        
        # Check whether it is the AMD ROCm build
        if hasattr(torch.version, 'hip') and torch.version.hip:
            # AMD ROCm PyTorch
            return "AMD", f"rocm{torch.version.hip}" if torch.version.hip else "rocm"
        
        # Check whether CUDA (NVIDIA) is supported
        elif torch.cuda.is_available():
            cuda_version = torch.version.cuda
            return "GPU", f"cu{cuda_version.replace('.', '')}" if cuda_version else "unknown"
        
        # Check whether MPS (Apple Silicon Metal) is supported
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "Metal", "mps"
        
        else:
            # CPU build
            torch_version = torch.__version__
            if '+cpu' in torch_version or 'cpu' in torch_version:
                return "CPU", "cpu"
            else:
                # Possibly the CPU build, without an explicit marker
                return "CPU", "cpu"
    except ImportError:
        # PyTorch is not installed
        return None, None

def get_dependency_group():
    """Get the matching pyproject dependency group"""
    torch_type, variant = detect_torch_type()
    
    if torch_type == "GPU":
        return "cuda12.6" if (variant or "").startswith("cu12") else "cuda13.0"
    elif torch_type == "AMD":
        return "rocm7.2.1"
    elif torch_type == "Metal":
        return "metal"
    elif torch_type == "CPU":
        return "cpu"
    else:
        # PyTorch is not installed, so it cannot be determined
        return None


def get_requirements_file():
    """For old callers; it now returns the dependency group name."""
    return get_dependency_group()

if __name__ == "__main__":
    torch_type, variant = detect_torch_type()
    
    if torch_type:
        print(f"检测到 PyTorch 类型: {torch_type}")
        if variant:
            print(f"变体: {variant}")
        
        group = get_dependency_group()
        print(f"对应的依赖组: {group}")
        
        # --file-only is kept as an alias for old callers
        if len(sys.argv) > 1 and sys.argv[1] in ("--group-only", "--file-only"):
            print(group, end="")
    else:
        print("未检测到 PyTorch，无法确定版本类型")
        print("将在安装依赖时重新选择")
        sys.exit(1)
