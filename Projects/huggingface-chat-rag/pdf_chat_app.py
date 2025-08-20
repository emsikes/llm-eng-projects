from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline
from huggingface_hub import login, notebook_login
import bitsandbytes as bnb
import torch

import gradio as gr
import pypdf
from IPython.display import display, Markdown
from pathlib import Path
import requests


available_models = {
    "Llama 3.2": "unsloth/Llama-3.2-3B-Instruct",
    "Microsoft Phi-4 Mini": "microsoft/Phi-4-mini-instruct",
    "Google Gemma 3": "unsloth/gemma-3-4b-it-GGUF"
}


current_model_id = None
current_pipeline = None
print(f"Models available to choose from: {list(available_models.keys())}")


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
    )




