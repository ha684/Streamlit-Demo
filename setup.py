import os
import re
import subprocess
import sys
from typing import List
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop

def get_cuda_version_from_toolkit():
    try:
        print("Checking CUDA version from toolkit...")
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        version_match = re.search(r'release (\d+\.\d+)', nvcc_output)
        if version_match:
            print(f"Detected CUDA version: {version_match.group(1)}")
            return version_match.group(1)
        print("No CUDA version detected.")
        return None
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error detecting CUDA version: {e}")
        return None

def get_pytorch_wheel_cuda_version(cuda_version):
    if cuda_version:
        version_str = cuda_version.replace('.', '')
        if len(version_str) == 2:
            version_str += '0'
        print(f"PyTorch CUDA version required: cu{version_str}")
        return f"cu{version_str}"
    print("CUDA version not found.")
    return None

def check_torch_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA is already available in PyTorch.")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            return True
    except ImportError:
        print("PyTorch not installed; proceeding with CUDA checks.")

    cuda_version = get_cuda_version_from_toolkit()
    
    if not cuda_version:
        print("NVIDIA CUDA Toolkit is not installed!")
        print("Please install the CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
        sys.exit(1)
    
    pytorch_cuda_version = get_pytorch_wheel_cuda_version(cuda_version)
    try:
        import torch
        current_cuda_version = f"cu{torch.version.cuda.replace('.', '')}" if torch.version.cuda else None
        if not torch.cuda.is_available() or (current_cuda_version and current_cuda_version != pytorch_cuda_version):
            print(f"Installing PyTorch with CUDA {cuda_version} support...")
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url",
                f"https://download.pytorch.org/whl/{pytorch_cuda_version}"
            ])
            import torch
            if not torch.cuda.is_available():
                print("PyTorch CUDA support installation failed!")
                sys.exit(1)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError:
        print(f"Installing PyTorch with CUDA {cuda_version} support...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url",
            f"https://download.pytorch.org/whl/{pytorch_cuda_version}"
        ])
        return check_torch_cuda()

def process_requirement(req):
    print(f"Processing requirement: {req}")
    if req.startswith('einops'):
        return 'einops==0.5.0'  
    strict_versions = ['vietocr']
    for pkg in strict_versions:
        if req.startswith(pkg):
            return req
    return req.replace('==', '>=')

def get_version():
    return "0.1.0"

def get_requires():
    with open("requirements.txt", encoding="utf-8") as f:
        print("Reading requirements.txt...")
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return [process_requirement(req) for req in lines]
    
def install_git_dependencies():
    if os.path.exists("requirements_git.txt"):
        print("\n=== Installing Git Dependencies ===")
        with open("requirements_git.txt") as f:
            git_deps = [line.strip() for line in f if line.strip()]
        
        for dep in git_deps:
            print(f"Installing Git dependency: {dep}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", dep])
                print(f"Successfully installed {dep}")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {dep}: {e}")
                raise

class PostDevelopCommand(develop):
    def run(self):
        print("Running post-develop command...")
        develop.run(self)
        install_git_dependencies()

class PostInstallCommand(install):
    def run(self):
        print("Running post-install command...")
        install.run(self)
        install_git_dependencies()

def main():
    print("Starting setup process...")
    check_torch_cuda()
    
    setup(
        name="smart-ocr",
        version=get_version(),
        author="PhanHa",
        author_email="phanha6844@gmail.com",
        description="Smart OCR Application with YOLO-based layout detection and multiple OCR options",
        long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
        long_description_content_type="text/markdown",
        keywords=["OCR", "Machine Learning", "Computer Vision", "YOLO", "PaddleOCR"],
        license="Apache 2.0 License",
        url="https://github.com/ha684/Demo_OCR",
        packages=find_packages(),
        python_requires=">=3.8.0",
        install_requires=get_requires(),
        extras_require={"dev": ["pytest", "black", "isort", "flake8"]},
        cmdclass={'install': PostInstallCommand, 'develop': PostDevelopCommand},
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )

if __name__ == "__main__":
    main()
