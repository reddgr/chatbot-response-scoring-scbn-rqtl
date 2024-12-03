
import sys
import os
import torch
import transformers

def check_env(colab=False, use_dotenv=False, dotenv_path=None):
    # Checking versions and GPU availability:
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device available")

    # Checks HuggingFace token
    if use_dotenv:
        print("Retrieved HuggingFace token(s) from .env file")
        from dotenv import load_dotenv
        load_dotenv("C:/apis/.env") # path to your dotenv file
        hf_token = os.getenv("HF_TOKEN")
        hf_token_write = os.getenv("HF_TOKEN_WRITE") # Only used for updating the Reddgr dataset (privileges needed)
    elif colab:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        hf_token_write = userdata.get('HF_TOKEN_WRITE')
    else:
        print("Retrieved HuggingFace token(s) from environment variables")
        hf_token = os.environ.get("HF_TOKEN")
        hf_token_write = os.environ.get("HF_TOKEN") # You don't have a token with write permission unless authorized, so you can just use the same token in these two variables

    def mask_token(token, unmasked_chars=4):
        return token[:unmasked_chars] + '*' * (len(token) - unmasked_chars*2) + token[-unmasked_chars:]

    if hf_token is None:
        print("HF_TOKEN not found in the provided .env file" if use_dotenv else "HF_TOKEN not found in the environment variables")
    if hf_token_write is None:
        print("HF_TOKEN_WRITE not found in the provided .env file" if use_dotenv else "HF_TOKEN_WRITE not found in the environment variables")

    masked_hf_token = mask_token(hf_token) if hf_token else None
    masked_hf_token_write = mask_token(hf_token_write) if hf_token_write else None

    if masked_hf_token:
        print(f"Using HuggingFace token: {masked_hf_token}")
    if masked_hf_token_write:
        print(f"Using HuggingFace write token: {masked_hf_token_write}")

    return hf_token, hf_token_write





def check_env_ORIG(colab=False, use_dotenv=False):
    # Checking versions and GPU availability:
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device available")

    # Checks HuggingFace token
    if use_dotenv:
        print("Using .env file to store HuggingFace token(s)")
        from dotenv import load_dotenv
        load_dotenv("C:/apis/.env") # path to your dotenv file
        hf_token = os.getenv("HF_TOKEN")
        hf_token_write = os.getenv("HF_TOKEN_WRITE") # Only used for updating the Reddgr dataset (privileges needed)
    elif colab:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        hf_token_write = userdata.get('HF_TOKEN_WRITE')
    else:
        print("Using environment variables to store HuggingFace token(s)")
        hf_token = os.environ.get("HF_TOKEN")
        hf_token_write = os.environ.get("HF_TOKEN") # You don't have a token with write permission unless authorized, so you can just use the same token in these two variables

    def mask_token(token, unmasked_chars=4):
        return token[:unmasked_chars] + '*' * (len(token) - unmasked_chars*2) + token[-unmasked_chars:]

    if hf_token is None:
        raise ValueError("HF_TOKEN not found in the provided .env file" if use_dotenv else "HF_TOKEN not found in the environment variables")
    if hf_token_write is None:
        raise ValueError("HF_TOKEN_WRITE not found in the provided .env file" if use_dotenv else "HF_TOKEN_WRITE not found in the environment variables")

    masked_hf_token = mask_token(hf_token)
    masked_hf_token_write = mask_token(hf_token_write)

    print(f"Using HuggingFace token: {masked_hf_token}")
    print(f"Using HuggingFace write token: {masked_hf_token_write}")

    return hf_token, hf_token_write