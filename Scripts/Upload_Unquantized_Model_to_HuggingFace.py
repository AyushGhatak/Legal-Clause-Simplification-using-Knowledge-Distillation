!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 -q
!pip install bitsandbytes==0.43.3 -q
!pip install transformers==4.44.0 peft==0.12.0 accelerate==0.34.0 -q


import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Path where checkpoint files / LoRA adapter weights are currently stored
LORA_ADAPTER_PATH = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/legal_chatbot_qwen_lora_model"

# The local folder where unified standalone model would reside
OUTPUT_DIRECTORY = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/merged_qwen_3b"

def merge_lora_to_base():
    print("Initializing model merge process...")

    # Determine execution precision based on hardware availability
    # FP16 is preferred on GPUs; standard system CPUs handle FP32 cleaner
    if torch.cuda.is_available():
        compute_dtype = torch.float16
        device_map = "auto"
        print("Detected CUDA GPU. Running merge operation in float16 precision.")
    else:
        compute_dtype = torch.float32
        device_map = {"": "cpu"}
        print("Running on CPU instance. Shifting to float32 processing precision...")

    # Load the base Qwen model
    # NOTE: Do NOT use quantization_config or load_in_4bit here.
    # We must load it unquantized to perform clean matrix addition!
    print(f"Loading base architecture: {BASE_MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )

    # Load and configure the companion tokenizer asset
    print("Loading tokenizer configuration...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # Bind fine-tuned delta adapter weights onto the base model skeleton
    print(f"Wrapping custom LoRA adapters from: {LORA_ADAPTER_PATH}...")
    model_with_adapter = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH,
        device_map=device_map
    )

    # Execute the math matrix combination step
    print("Fusing weights permanently (merge_and_unload)...")
    merged_model = model_with_adapter.merge_and_unload()

    # Write out the final standalone model directory files
    print(f"Saving newly fused model down to target path: {OUTPUT_DIRECTORY}...")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    merged_model.save_pretrained(OUTPUT_DIRECTORY, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIRECTORY)

    print("\nDone! Your unified model weights are fully compiled and ready for Phase 2!")

if __name__ == "__main__":
    merge_lora_to_base()


import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "ayushhh1662309"

REPO_NAME = "qwen2.5-3b-legal-merged"

# The local folder containing safe_serialization tensors and tokenizer
LOCAL_FOLDER_PATH = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/merged_qwen_3b"

HF_TOKEN = "hf_JcBsZANlSzO***********************" # Token with write access

def upload_model_to_hf():
    # Combined user and repo name to create the full identifier
    full_repo_id = f"{HF_USERNAME}/{REPO_NAME}"

    print(f"Initializing connection to Hugging Face Hub...")
    api = HfApi(token=HF_TOKEN)

    # Safely creating the target model repository online
    try:
        print(f"Creating repository: {full_repo_id}...")
        create_repo(
            repo_id=full_repo_id,
            repo_type="model",
            private=False,
            token=HF_TOKEN,
            exist_ok=True   # If the repository already exists, don't crash
        )
        print("Repository verified/created successfully.")
    except Exception as e:
        print(f"Repository creation warning: {e}")

    # Uploading the entire folder content in a single operation
    print(f"Uploading folder '{LOCAL_FOLDER_PATH}' to '{full_repo_id}'...")
    print("This may take a few minutes...")

    try:
        api.upload_folder(
            folder_path=LOCAL_FOLDER_PATH,
            repo_id=full_repo_id,
            repo_type="model",
            commit_message="Initial commit: Fused Qwen2.5 3B Legal Contract Simplifier weights",
        )
        print(f"\nSuccess! Your model is live at: https://huggingface.co/{full_repo_id}")
    except Exception as e:
        print(f"\nUpload failed: {e}")

if __name__ == "__main__":
    # Quick check to ensure the local folder exists before triggering network lines
    if not os.path.exists(LOCAL_FOLDER_PATH):
        print(f" Error: Local folder '{LOCAL_FOLDER_PATH}' not found. Please run your merge script first!")
    else:
        upload_model_to_hf()


