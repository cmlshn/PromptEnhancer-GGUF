"""
Copyright (c) 2025 Tencent. All Rights Reserved.
Licensed under the Tencent Hunyuan Community License Agreement.

GGUF version of PromptEnhancer using llama-cpp-python for quantized models.
"""

import re
import os
import time
import logging
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama-cpp-python is required for GGUF inference. "
        "Install it with: pip install llama-cpp-python[cuda]"
    )

def replace_single_quotes(text):
    """
    Replace single quotes within words with double quotes, and convert
    curly single quotes to curly double quotes for consistency.
    """
    pattern = r"\B'([^']*)'\B"
    replaced_text = re.sub(pattern, r'"\1"', text)
    replaced_text = replaced_text.replace("'", """)
    replaced_text = replaced_text.replace("'", """)
    return replaced_text

class PromptEnhancerGGUF:
    
    def __init__(
        self, 
        model_path: str, 
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,  # -1 means use all GPU layers
        verbose: bool = False
    ):
        """
        Initialize the PromptEnhancerGGUF class with GGUF model.

        Args:
            model_path (str): Path to the GGUF model file.
            n_ctx (int): Context window size.
            n_gpu_layers (int): Number of layers to offload to GPU (-1 for all).
            verbose (bool): Enable verbose logging from llama.cpp.
        """
        # Lazy logging setup
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info(f"Loading GGUF model from: {model_path}")
        
        # Initialize the GGUF model with explicit GPU settings
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            # GPU acceleration settings
            n_batch=512,
            use_mmap=True,
            use_mlock=False,
            # Force GPU usage
            main_gpu=0,
            tensor_split=None,
        )
        
        self.logger.info("GGUF model loaded successfully")

    def predict(
        self,
        prompt_cot: str,
        sys_prompt: str = "You are an expert at enhancing image generation prompts. Rewrite the user's prompt to be more detailed and descriptive while keeping the original intent. Make it clear, well-structured, and suitable for image generation.",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate a rewritten prompt using the GGUF model.

        Args:
            prompt_cot (str): The original prompt to be rewritten.
            sys_prompt (str): System prompt to guide the rewriting.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling parameter.
            max_new_tokens (int): Maximum number of new tokens to generate.
            **kwargs: Additional arguments for llama.cpp.

        Returns:
            str: The rewritten prompt, or the original if generation fails.
        """
        org_prompt_cot = prompt_cot
        
        try:
            # Simplified prompt format - try without chat template first
            full_prompt = f"System: {sys_prompt}\n\nUser: {org_prompt_cot}\n\nAssistant:"
            
            
            # Generate response with timeout and simpler parameters
            response = self.llm(
                full_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["User:", "System:", "\n\n"],
                echo=False,
                stream=False,
                **kwargs
            )
            
            # Extract the generated text
            output_text = response['choices'][0]['text'].strip()
            
            # Parse the output to extract the rewritten prompt
            if output_text:
                # Handle potential think tags or other formatting
                if "think>" in output_text and output_text.count("think>") == 2:
                    prompt_cot = output_text.split("think>")[-1].strip()
                else:
                    prompt_cot = output_text
                
                # Clean up the prompt
                prompt_cot = replace_single_quotes(prompt_cot)
            else:
                prompt_cot = org_prompt_cot
                self.logger.warning("Empty response; using the original prompt")

        except Exception as e:
            prompt_cot = org_prompt_cot
            self.logger.exception("Re-prompting failed; using the original prompt")

        return prompt_cot

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "n_ctx": self.llm.n_ctx(),
            "n_vocab": self.llm.n_vocab(),
            "model_path": self.llm.model_path if hasattr(self.llm, 'model_path') else "Unknown"
        }

if __name__ == "__main__":
    # Auto-detect model path or use environment variable
    model_path = os.environ.get('GGUF_MODEL_PATH')
    
    if not model_path:
        # Default to Q8_0 model in models folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        models_dir = os.path.join(repo_root, "models")
        
        # Look for Q8_0 model first, then any GGUF file
        q8_model = os.path.join(models_dir, "PromptEnhancer-32B.Q8_0.gguf")
        
        if os.path.exists(q8_model):
            model_path = q8_model
        else:
            # Look for any GGUF file in models directory
            if os.path.exists(models_dir):
                gguf_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
                if gguf_files:
                    model_path = os.path.join(models_dir, gguf_files[0])
                    print(f"Q8_0 model not found, using: {gguf_files[0]}")
        
        if not model_path:
            print("Error: No GGUF model found in models/ directory")
            print("Please download a model or set GGUF_MODEL_PATH environment variable")
            print("Usage: GGUF_MODEL_PATH='/path/to/your/model.gguf' python inference/prompt_enhancer_gguf.py")
            exit(1)
    
    # Initialize the GGUF prompt enhancer
    try:
        prompt_enhancer_cls = PromptEnhancerGGUF(
            model_path=model_path,
            n_ctx=1024,  # Even smaller context for faster inference
            n_gpu_layers=-1,  # Use all GPU layers
            verbose=False  # Minimize verbose output
        )
        
        # Print model info
        model_info = prompt_enhancer_cls.get_model_info()
        print(f"Model Info: {model_info}")
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please download a GGUF model from: https://huggingface.co/mradermacher/PromptEnhancer-32B-GGUF")
        print("Recommended for RTX 5090 (32GB VRAM):")
        print("  - Q6_K (27GB) - Best quality that fits")
        print("  - Q8_0 (35GB) - Highest quality (might be tight)")
        print("  - Q4_K_M (20GB) - Good balance")
        exit(1)
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install llama-cpp-python[cuda]")
        exit(1)

    # Test prompts
    test_list_en = [
        "woman in jungle",
        "sports car in monaco", 
        "a cat in a room",
        "digital art explosion of the colors"
    ]

    print("Testing English prompts with GGUF model:")
    print("=" * 50)
    
    for i, item in enumerate(test_list_en, 1):
        print(f"\n[{i}/{len(test_list_en)}] User Prompt: {item}")
        print("-" * 40)
        
        time_start = time.time()
        result = prompt_enhancer_cls.predict(
            item, 
            max_new_tokens=512,  # Fixed token count for consistent timing
            temperature=0.3,
            top_p=0.9,
            # Add min_tokens to ensure consistent length
        )
        time_end = time.time()
        
        print(f"Enhanced Prompt: {result}")
        print(f"Time cost: {time_end - time_start:.2f}s")
        print("~" * 50)
