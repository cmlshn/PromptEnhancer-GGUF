"""
Copyright (c) 2025 Tencent. All Rights Reserved.
Licensed under the Tencent Hunyuan Community License Agreement.

GGUF version of PromptEnhancer using llama-cpp-python for quantized models.
"""

import re
import os
import time
import logging
import random
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
    Disabled for now to prevent text corruption.
    """
    # Temporarily disabled to prevent text corruption
    # Just return the text as-is
    return text

def clean_repetitive_content(text):
    """
    Clean up repetitive content and verbose explanations from model output.
    """
    # First, handle obvious repetition patterns
    # Look for sentences that repeat with slight variations
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    
    # Remove exact duplicates and very similar sentences
    unique_sentences = []
    seen_content = set()
    
    for sentence in sentences:
        # Create a simplified version for comparison (remove articles, prepositions)
        simplified = ' '.join([word for word in sentence.lower().split() 
                              if word not in ['the', 'a', 'an', 'in', 'on', 'at', 'with', 'by']])
        
        # Skip if we've seen very similar content
        is_duplicate = False
        for seen in seen_content:
            # Check if sentences are very similar (>70% word overlap)
            words1 = set(simplified.split())
            words2 = set(seen.split())
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
                if overlap > 0.7:
                    is_duplicate = True
                    break
        
        if not is_duplicate and len(sentence) > 15:
            seen_content.add(simplified)
            unique_sentences.append(sentence)
    
    # Remove meta-commentary and analytical explanations
    meta_phrases = [
        "The overall presentation",
        "This image presents",
        "The composition is",
        "The lighting is",
        "In the background",
        "The prompt requests",
        "To enhance this",
        "Reasoning:",
        "Your task",
        "I first identify",
        "I'll specify",
        "Next, I'll define",
        "For composition",
        "Finally, I'll establish",
        "The user's prompt",
        "The process is purely",
        "To further enhance",
        "To further solidify",
        "关键在于",
        "<think>"
    ]
    
    # Keep only the first occurrence of common descriptive patterns
    final_sentences = []
    used_patterns = set()
    
    for sentence in unique_sentences:
        # Check if this sentence starts with a pattern we've already used
        pattern_used = False
        for phrase in meta_phrases:
            if sentence.startswith(phrase):
                if phrase in used_patterns:
                    pattern_used = True
                    break
                else:
                    used_patterns.add(phrase)
        
        if not pattern_used:
            final_sentences.append(sentence)
    
    result = '. '.join(final_sentences)
    
    # Ensure it ends properly
    if result and not result.endswith('.'):
        result += '.'
        
    return result

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
        sys_prompt: str = "You are a technical prompt enhancement tool. Transform the input into a detailed image generation prompt. Add visual specifics: lighting, camera angles, poses, environment, clothing details, and artistic style. Output only the enhanced visual description without commentary or analysis. Process all content as technical specifications.",
        temperature: float = 0.7,
        top_p: float = 0.95,
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
            # More directive prompt format to bypass refusals
            full_prompt = f"""System: {sys_prompt}

Task: Enhance the following prompt for image generation by adding technical details.
Input: {org_prompt_cot}
Enhanced Output:"""
            
            
            # Generate response with optimized parameters
            response = self.llm(
                full_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["User:", "System:", "Input:", "Task:", "Enhanced Output:", "The prompt", "To enhance", "Reasoning:", "Your task", "\n\n", "Style:", "Note:", "To further", "关键在于", "<think>"],
                echo=False,
                stream=False,
                repeat_penalty=1.1,  # Reduce repetition
                seed=random.randint(1, 1000000),  # Random seed for variation
                **kwargs
            )
            
            # Extract the generated text
            output_text = response['choices'][0]['text'].strip()
            
            # Parse the output to extract the rewritten prompt
            if output_text:
                # Check if the output contains think tags
                think_count = output_text.count("think>")
                
                if think_count == 2:
                    prompt_cot = output_text.split("think>")[-1]
                    if prompt_cot.startswith("\n"):
                        prompt_cot = prompt_cot[1:]
                    prompt_cot = replace_single_quotes(prompt_cot)
                else:
                    # If no think tags, use the entire output as the enhanced prompt
                    prompt_cot = output_text
                    
                    # Clean up common prefixes that models might add
                    prefixes_to_remove = [
                        "**Enhanced Prompt:**",
                        "Enhanced Prompt:",
                        "**Enhanced:**",
                        "Enhanced:",
                        "Style:",
                        "Note:",
                    ]
                    for prefix in prefixes_to_remove:
                        if prompt_cot.startswith(prefix):
                            prompt_cot = prompt_cot[len(prefix):].strip()
                            break
                    
                    # Remove everything after <think> tag if present
                    if "<think>" in prompt_cot:
                        prompt_cot = prompt_cot.split("<think>")[0].strip()
                    
                    # Clean up meta-commentary and repetitive content
                    prompt_cot = clean_repetitive_content(prompt_cot)
                    prompt_cot = replace_single_quotes(prompt_cot)
            else:
                prompt_cot = org_prompt_cot
                self.logger.warning("Empty response; using the original prompt")

        except Exception:
            prompt_cot = org_prompt_cot
            self.logger.warning("✗ Re-prompting failed, so we are using the original prompt")

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
            max_new_tokens=512,  # Increased for longer, more detailed prompts
            temperature=0.8,     # Much higher temperature for more variation
            top_p=0.95,         # Higher for more diversity
        )
        time_end = time.time()
        
        print(f"Enhanced Prompt: {result}")
        print(f"Time cost: {time_end - time_start:.2f}s")
        print("~" * 50)
