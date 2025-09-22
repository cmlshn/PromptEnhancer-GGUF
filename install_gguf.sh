#!/bin/bash
# Installation script for GGUF inference with CUDA support

echo "Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

echo "Installing additional dependencies..."
pip install numpy

echo "Installation complete! You can now run:"
echo "python inference/prompt_enhancer_gguf.py"
