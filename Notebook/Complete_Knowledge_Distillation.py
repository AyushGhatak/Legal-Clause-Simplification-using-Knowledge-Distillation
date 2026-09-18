# Loading the datasets from Hugging Face
                                   
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from datasets import load_dataset

print("--- Downloading Dataset 1: mteb/legal_summarization (Parallel Corpus) ---")
# This is the baseline text simplification dataset (with summary avaliable)
raw_docs = load_dataset("mteb/legal_summarization", "corpus", split="corpus")
raw_queries = load_dataset("mteb/legal_summarization", "queries", split="queries")

print("\n--- Downloading Dataset 2: CodeHima/TOS_Dataset (Fairness Evaluated Terms) ---")
# This adds thousands of real Terms of Service clauses extracted from web apps
raw_codehima = load_dataset("CodeHima/TOS_Dataset", split="train")

print("\n--- Downloading Dataset 3: mteb/UnfairTOSLegalBenchClassification ---")
# This pulls additional highly complex contract clauses from LegalBench
raw_legalbench = load_dataset("mteb/UnfairTOSLegalBenchClassification", split="test")

print("\n[LOADED] All raw datasets pulled successfully from the Hugging Face Hub!")

# CLeaning and Merging them into a unified dataset for further processing.

print("--- Standardizing and Compiling Master Dataset Matrix ---")
unified_records = []

# PARSE DATASET 1: mteb/legal_summarization
df_docs = pd.DataFrame(raw_docs)
df_queries = pd.DataFrame(raw_queries)

matrix_range = min(len(df_docs), len(df_queries))
for idx in range(matrix_range):
    legalese = str(df_docs.iloc[idx]["text"]).strip()
    summary = str(df_queries.iloc[idx]["text"]).strip() # Aligned summaries

    if len(legalese) > 30:
        unified_records.append({
            "source_dataset": "legal_summarization",
            "raw_legalese": legalese,
            "human_summary": summary # This dataset includes paired human summaries
        })

# PARSE DATASET 2: CodeHima/TOS_Dataset
for row in raw_codehima:
    legalese = str(row.get("sentence", "")).strip()
    if len(legalese) > 30:
        unified_records.append({
            "source_dataset": "CodeHima_TOS",
            "raw_legalese": legalese,
            "human_summary": "None" # These will receive summaries during our distillation pass!
        })

# PARSE DATASET 3: mteb/UnfairTOSLegalBenchClassification
# NOTE: We intentionally target split="test" because the MTEB benchmark configuration
# restricts the train split to 9 rows, while storing all 2,057 core contract clauses in the test split.
for row in raw_legalbench:
    legalese = str(row.get("text", "")).strip()
    if len(legalese) > 30:
        unified_records.append({
            "source_dataset": "LegalBench_UnfairTOS",
            "raw_legalese": legalese,
            "human_summary": "None" # Will be filled dynamically during the teacher distillation step!
        })


df_master_tos = pd.DataFrame(unified_records)

# Drop any exact duplicate clauses that exist across the different benchmark sets
df_master_tos = df_master_tos.drop_duplicates(subset=["raw_legalese"], keep="first").reset_index(drop=True)

print(f"\n[SUCCESS] Unified dataset compilation complete!")
print(f" -> Total Unique Contract Clauses Compiled: {len(df_master_tos)}")
print(f" -> Samples from mteb/legal_summarization: {len(df_master_tos[df_master_tos['source_dataset']=='legal_summarization'])}")
print(f" -> Samples from CodeHima/TOS_Dataset: {len(df_master_tos[df_master_tos['source_dataset']=='CodeHima_TOS'])}")
print(f" -> Samples from LegalBench_UnfairTOS: {len(df_master_tos[df_master_tos['source_dataset']=='LegalBench_UnfairTOS'])}")

# Save to Colab environment files
df_master_tos.to_csv("/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/master_compiled_legal_dataset.csv", index=False)
print("\nMaster dataset successfully written locally to: /content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/master_compiled_legal_dataset.csv")

# VERIFICATION PREVIEW
print("="*80)
print(" MASTER DATASET PREVIEW CHECK (Row 500 Sample):")
print("="*80)
print(f"SOURCE TRAJECTORY: {df_master_tos.iloc[500]['source_dataset']}")
print(f"RAW CONTRACT LEGALESE:\n{df_master_tos.iloc[500]['raw_legalese'][:350]}...")
print("="*80)

# Knowlege Distillation

# Teacher Model - accesing using serverless inference API

!pip install groq

import os
import re
import time
import pandas as pd
from groq import Groq
from tqdm import tqdm

print("--- Step 1: Loading Master Unified Dataset ---")
master_path = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/master_compiled_legal_dataset.csv"
output_path = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/latest_distilled_legal_dataset.csv"
state_tracker_path = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/latest_distilled_state_tracker.csv"

df_master = pd.read_csv(master_path)

# --- STRATIFIED SAMPLING ENGINE ---
sample_limit = 450
rows_per_dataset = sample_limit // 3
sources = ["legal_summarization", "CodeHima_TOS", "LegalBench_UnfairTOS"]
sampled_dfs = []

for src in sources:
    df_src = df_master[df_master["source_dataset"] == src]
    count = min(len(df_src), rows_per_dataset)
    sampled_dfs.append(df_src.sample(n=count, random_state=42, replace=False))

df_target_pool = pd.concat(sampled_dfs).reset_index(drop=True)

# --- STATE RECOVERY ---
if os.path.exists(output_path) and os.path.exists(state_tracker_path):
    print("-> Found existing progress files! Restoring state tracking metrics...")
    df_existing_data = pd.read_csv(output_path)
    df_existing_state = pd.read_csv(state_tracker_path)
    distilled_records = df_existing_data.to_dict("records")
    completed_clauses = set(df_existing_state["raw_legalese"].tolist())
    print(f"-> Resuming execution. Already processed {len(completed_clauses)} / {sample_limit} rows.")
else:
    print("-> No previous progress found. Starting fresh run...")
    distilled_records = []
    completed_clauses = set()

df_todo = df_target_pool[~df_target_pool["raw_legalese"].isin(completed_clauses)]
print(f"-> Unique rows left to process in this session: {len(df_todo)}")

# Initialize Groq Client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_NcSM8V8TmntEKzttl1Ve*********************************") # Groq AI read access token
client = Groq(api_key=GROQ_API_KEY)

# Strict system instruction forcing clean output tags
SYSTEM_INSTRUCTION = "You are a precise machine-readable data generator. You must output the tags <Reasoning>: and <Summary>: exactly as requested. Never output preamble conversational text."

# --- COGNITIVE CHAIN-OF-THOUGHT PROMPT ENGINE ---
PROMPT_GENERATE_ALL = """[TASK]
Deconstruct the provided legal contract clause step-by-step to show your cognitive thinking trace, then generate a highly structured plain-English markdown list summary.

[OUTPUT FORMAT]
<Reasoning>: Use the following 3-step cognitive breakdown process explicitly:
1. IDENTIFY: State exactly what type of clause this is and its primary legal objective.
2. TRANSLATE: Isolate dense, confusing, or ambiguous legal jargon terms and define what they mean to an everyday person.
3. COGNITIVE TRACE: Explain step-by-step how you parse the logic or conditions hidden in the syntax to arrive at a clean simplification.

<Summary>: Generate a highly comprehensive markdown list simplifying every single distinct rule or section in the text based on your reasoning above.

[SUMMARY RULES]
- Every single bullet point MUST start with a plain '*' character. Do NOT use markdown bolding (**) anywhere.
- Format the start of each bullet point exactly like this: '* Topic Name: Description...'
- Provide 1-2 impactful plain English sentences per bullet point.

[DATA]
Clause: {legalese} {metadata_hint}"""

print(f"\n--- Step 2: Processing Remaining Distillation Matrix ---")

halt_run = False

for idx, row in tqdm(df_todo.iterrows(), total=len(df_todo)):
    if halt_run:
        break

    legalese_text = str(row["raw_legalese"]).strip()
    dataset_source = str(row["source_dataset"])
    human_summary = str(row["human_summary"]).strip()

    # Anchor context against human summaries when available to avoid any information loss
    if dataset_source == "legal_summarization" and human_summary != "None" and human_summary != "":
        hint_text = f"\n\n[REFERENCE GROUND TRUTH SUMMARIZATION]\nUse this text to ensure zero information loss, but reformat it completely to match our rules:\n{human_summary}"
    else:
        hint_text = ""

    user_content = PROMPT_GENERATE_ALL.format(legalese=legalese_text, metadata_hint=hint_text)

    success = False
    retries = 0
    generation_temperature = 0.0  # Dropped to 0.0 for absolute structure adherence

    while not success and retries < 2:
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=768, # Increased window size to handle reasoning trace + comprehensive summaries without clipping
                temperature=generation_temperature,
            )

            raw_output = response.choices[0].message.content
            if not raw_output:
                retries += 1
                continue

            raw_output = raw_output.strip()

            # Normalize structural anomalies instantly
            normalized_output = raw_output.replace("**<Reasoning>:**", "<Reasoning>:").replace("**<Summary>:**", "<Summary>:")
            normalized_output = normalized_output.replace("**Reasoning**:", "<Reasoning>:").replace("**Summary**:", "<Summary>:")
            normalized_output = re.sub(r"```[a-zA-A]*\n|```", "", normalized_output).strip()

            # --- HYPER-RESILIENT PARSING REGEX ENGINE ---
            reasoning_match = re.search(r"(?:<Reasoning>|Reasoning)\s*[:\-–]?\s*(.*?)((?=<Summary>|Summary)|$)", normalized_output, re.IGNORECASE | re.DOTALL)
            summary_match = re.search(r"(?:<Summary>|Summary)\s*[:\-–]?\s*(.*)", normalized_output, re.IGNORECASE | re.DOTALL)

            extracted_reasoning = reasoning_match.group(1).strip() if reasoning_match else None
            extracted_summary = summary_match.group(1).strip() if summary_match else None

            if extracted_summary and extracted_reasoning:
                # Strip external wrapping artifacts safely
                clean_summary = extracted_summary.strip()
                clean_reasoning = extracted_reasoning.strip()

                # Remove the accidental bold bleed within the summary lines
                if "**" in clean_summary:
                    # Strip out bold marks everywhere, keeping the raw content intact
                    clean_summary = clean_summary.replace("**", "")

                # Ensure the block strictly starts with a valid markdown asterisk
                if not clean_summary.startswith("*"):
                    # Strip leading spaces/newlines/punctuation, then force prefix
                    clean_summary = re.sub(r"^[^a-zA-Z0-9*]*", "", clean_summary)
                    if not clean_summary.startswith("*"):
                        clean_summary = "* " + clean_summary

                # Fix broken structural formatting patterns like "* Topic Name **:"
                clean_summary = re.sub(r"\*\s*:\s*", ": ", clean_summary)

                success = True
            else:
                print(f"\n[PARSING FAIL] Row {idx}, Attempt {retries+1}: Missing reasoning or summary structure. Bumping temperature and retrying...")
                generation_temperature = 0.2
                retries += 1

            if success:
                # Re-verify that formatting rules are cleanly adhered to
                if "**" in clean_summary or not clean_summary.startswith("*"):
                     print(f"\n[CRITICAL WARNING] Row {idx} auto-healing failed to normalize formatting. Inspect output manually.")
                else:
                     print(f"\n[SUCCESS] Row {idx} distilled and cleanly formatted.")

                # Split output into two dedicated schema targets
                new_record = {
                    "raw_legalese": legalese_text,
                    "distilled_reasoning": clean_reasoning,
                    "distilled_summary": clean_summary,
                    "source_dataset": dataset_source
                }

                distilled_records.append(new_record)
                completed_clauses.add(legalese_text)

                df_new_row = pd.DataFrame([new_record])
                df_new_state = pd.DataFrame([{"raw_legalese": legalese_text}])

                file_exists = os.path.exists(output_path)
                state_exists = os.path.exists(state_tracker_path)

                df_new_row.to_csv(
                    output_path,
                    mode='a',
                    header=not file_exists,
                    index=False
                )

                df_new_state.to_csv(
                    state_tracker_path,
                    mode='a',
                    header=not state_exists,
                    index=False
                )

        except Exception as e:
            error_msg = str(e)
            if "tokens per day" in error_msg.lower() or "tpd" in error_msg.lower() or "429" in error_msg:
                print(f"\n[RATE CEILING DETECTED] Processing limit hit at row index {idx}. Stopping pipeline safely.")
                halt_run = True
                break

            print(f"\n[API EXCEPTION] Row Index {idx}, Attempt {retries+1}: {error_msg}")
            retries += 1
            time.sleep(5)

    if halt_run:
        break

print("\n" + "=" * 80)
print(f"[STATUS] Data collection phase complete/paused for this session.")
print(f"Total entries currently saved in dataset: {len(distilled_records)} / {sample_limit}")
print("=" * 80)

# Loading and Preparing the dataset for Student Model Fine Tuning

import pandas as pd
import json

df = pd.read_csv('/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/latest_distilled_legal_dataset.csv')
print(f"Before: {len(df)} rows")

# Drop the 9 known bad rows by their index - these rows were identified
bad_indices = [10, 41, 107, 112, 162, 173, 188, 240, 422]
df = df.drop(index=bad_indices).reset_index(drop=True)
print(f"After dropping bad rows: {len(df)} rows")

# Check format distribution
format_identify = df['distilled_reasoning'].str.contains('1. IDENTIFY', na=False).sum()
format_step     = df['distilled_reasoning'].str.contains('## Step 1', na=False).sum()
print(f"\n'1. IDENTIFY' format: {format_identify}")
print(f"'## Step 1' format:   {format_step}")

# Keep only the dominant format
df = df[df['distilled_reasoning'].str.contains('1. IDENTIFY', na=False)].reset_index(drop=True)
print(f"Final clean rows: {len(df)}")  # should be 434

# Verify all 3 steps present in every row
missing_translate     = df[~df['distilled_reasoning'].str.contains('2. TRANSLATE', na=False)]
missing_cognitive     = df[~df['distilled_reasoning'].str.contains('3. COGNITIVE', na=False)]
print(f"Missing TRANSLATE:     {len(missing_translate)}")
print(f"Missing COGNITIVE:     {len(missing_cognitive)}")

# Find and inspect the bad row
missing = df[
    ~df['distilled_reasoning'].str.contains('2. TRANSLATE', na=False) |
    ~df['distilled_reasoning'].str.contains('3. COGNITIVE', na=False)
]
print(f"Bad rows: {len(missing)}")
print(missing[['raw_legalese', 'distilled_reasoning']].to_string())

# Drop it
df = df[
    df['distilled_reasoning'].str.contains('2. TRANSLATE', na=False) &
    df['distilled_reasoning'].str.contains('3. COGNITIVE', na=False)
].reset_index(drop=True)

print(f"Final clean rows: {len(df)}")

# Save the cleaned distilled dataset
df.to_csv('/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/latest_distilled_legal_dataset.csv', index = False)

# Rechecking if everything is fine
df = pd.read_csv('/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/latest_distilled_legal_dataset.csv')
missing = df[
    ~df['distilled_reasoning'].str.contains('1. IDENTIFY', na=False) |
    ~df['distilled_reasoning'].str.contains('2. TRANSLATE', na=False) |
    ~df['distilled_reasoning'].str.contains('3. COGNITIVE', na=False)
]
print(f"Bad rows: {len(missing)}")
print(missing[['raw_legalese', 'distilled_reasoning']].to_string())

import pandas as pd

# Load newly generated distilled dataset
df_distilled = pd.read_csv("/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/latest_distilled_legal_dataset.csv")

print("--- Verifying Distilled Dataset Shape ---")
print(f"Total rows successfully generated: {len(df_distilled)}")
print("Columns available:", df_distilled.columns.tolist())

print("\n--- Raw Sample Extraction ---")
print("[INPUT LEGALESE]:\n", df_distilled.iloc[322]["raw_legalese"][:5000], "\n")
print("-" * 50)
print("[Reasoning Block]:\n", df_distilled.iloc[322]["distilled_reasoning"][:5000], "\n")
print("-" * 50)
print("[Summary Block]:\n", df_distilled.iloc[322]["distilled_summary"][:5000])

# Preparing the JSON file

import os
import json
import pandas as pd

# Loading the distilled dataset
dataset_path = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/latest_distilled_legal_dataset.csv"
save_path = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/latest_sft_chatbot_dataset.jsonl"

df = pd.read_csv(dataset_path)

print("--- Verifying Distilled Dataset Shape ---")
print(f"Total rows successfully loaded: {len(df)}")
print("Columns available:", df.columns.tolist())

# system message that instructs the student model to reason first, then summarize
system_message = """<persona>
    You are a professional legal analyst. You simplify complex legal documents into clear, plain English for non-lawyers.
    </persona>

    <methodology>
    Always follow exactly two phases. Never skip or merge them.

    PHASE 1 — Output under the tag '<Reasoning>:' — Do these three steps:
      STEP 1 - IDENTIFY: State the document type, jurisdiction if mentioned, and its legal purpose.
      STEP 2 - TRANSLATE: List every technical legal term and define it in plain English.
      STEP 3 - DECONSTRUCT: Trace the logic step-by-step. Expose all conditions, exceptions, and jurisdiction-specific rules.

    PHASE 2 — Output under the tag '<Summary>:' — Write a structured summary:
      - Write as a standalone document. Never copy or restate the Reasoning.
      - Use markdown headers: ### 1. Section Name
      - Every bullet MUST start with: * Topic Name: Description
      - Keep each bullet to 1-2 sentences maximum.
      - When a topic has multiple items (countries, data types, conditions), list EVERY one as indented sub-bullets:
          - Sub-item: Description
      - Never skip, merge, or omit any item from the source text.
    </methodology>

    <rules>
    1. COMPLETENESS: Every section, data type, jurisdiction, and condition in the source must appear explicitly.
    2. CONDITIONALS: Never turn a conditional into an absolute. If the source says "only in limited countries" or "depending on your settings", keep that condition in your output.
    3. DIRECTIONALITY: Always state who owes what to whom. Never reverse company obligations and user obligations.
    4. LIABILITY: Flag any "as-is", warranty disclaimers, or penalty assignments explicitly.
    5. READING LEVEL: Write at a 6th-grade reading level throughout.
    </rules>"""

jsonl_records = []

for idx, row in df.iterrows():
    # Construct the target sequence containing the CoT reasoning followed by the summary.
    # This trains the student model to reason internally before outputting the final user response.
    assistant_target = f"<Reasoning>:\n{row['distilled_reasoning']}\n\n<Summary>:\n{row['distilled_summary']}"

    # Build the standard ChatML formatting array
    chat_format = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": (
                "<task>\n"
                "Analyze and simplify the following legal text. Apply your full CoT methodology "
                "and produce a complete structured summary as specified in your instructions.\n"
                "</task>\n\n"
                f"<contract_clause>\n{row['raw_legalese']}\n</contract_clause>"
            )},
            {"role": "assistant", "content": assistant_target}
        ]
    }
    jsonl_records.append(chat_format)

# Write out to standard JSONL format
with open(save_path, "w", encoding="utf-8") as f:
    for record in jsonl_records:
        f.write(json.dumps(record) + "\n")

print(f"\nSuccessfully converted {len(jsonl_records)} rows into Chain-of-Thought ChatML format for chatbot training!")
print(f"Target file saved to location: {save_path}")

# Student Model
!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 -q
!pip install bitsandbytes==0.43.3 -q
!pip install transformers==4.44.0 peft==0.12.0 accelerate==0.34.0 -q

## Checking if Qwen 2.5 is working fine on a sample text!
import torch
import bitsandbytes as bnb
print(f"PyTorch: {torch.__version__}")
print(f"BNB:     {bnb.__version__}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit           = True,
    bnb_4bit_quant_type    = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config = bnb_config,
    device_map          = "auto",
    torch_dtype         = torch.float16,
)

# Test with longer, more natural text to get a meaningful loss reading
test_texts = [
    "The contract states that all parties must agree to the terms.",
    "This agreement is governed by the laws of the state of California.",
    "The user agrees not to reproduce or distribute any content.",
]

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    print(f"Loss: {outputs.loss.item():.4f} | Text: {text[:50]}")

model.eval()
inputs = tokenizer(
    "Simplify this legal clause: The party of the first part agrees to indemnify",
    return_tensors="pt"
).to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens = 200,
        temperature    = 0.7,
        do_sample      = True,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Fine Tuning (LoRA + QLoRA)

import os
import gc
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset

MAX_SEQ_LENGTH = 5120

dataset_input_path    = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/latest_sft_chatbot_dataset.jsonl"
drive_model_save_path = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/legal_chatbot_qwen_lora_model"

#  Load model in 4-bit via BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map          = "auto",
    torch_dtype         = torch.float16,
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing    = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
)

# Apply LoRA
lora_config = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize dataset with masking
raw_dataset = load_dataset("json", data_files=dataset_input_path, split="train")

assistant_header     = "<|im_start|>assistant\n"
assistant_header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)

def tokenize_and_mask(examples):
    batch_input_ids, batch_attention_mask, batch_labels = [], [], []

    for msg_list in examples["messages"]:
        conversation = [
            {"role": "system",    "content": msg_list[0]["content"]},
            {"role": "user",      "content": msg_list[1]["content"]},
            {"role": "assistant", "content": msg_list[2]["content"]},
        ]
        full_ids = tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False
        )
        if isinstance(full_ids[0], list):
            full_ids = full_ids[0]

        labels = list(full_ids)
        response_start_idx = len(full_ids)

        for i in range(len(full_ids) - len(assistant_header_ids) + 1):
            if full_ids[i: i + len(assistant_header_ids)] == assistant_header_ids:
                response_start_idx = i + len(assistant_header_ids)
                break

        for idx in range(response_start_idx):
            labels[idx] = -100

        attention_mask = [1] * len(full_ids)

        if len(full_ids) > MAX_SEQ_LENGTH:
            full_ids       = full_ids[:MAX_SEQ_LENGTH]
            attention_mask = attention_mask[:MAX_SEQ_LENGTH]
            labels         = labels[:MAX_SEQ_LENGTH]

        batch_input_ids.append(full_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids":      batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels":         batch_labels,
    }

tokenized_dataset = raw_dataset.map(tokenize_and_mask, batched=True)

# Verify masking
sample_ids    = torch.tensor(tokenized_dataset[0]["input_ids"])
sample_labels = torch.tensor(tokenized_dataset[0]["labels"])
non_masked    = (sample_labels != -100).sum().item()
print(f"\nTotal tokens: {len(sample_ids)} | Trained on: {non_masked}")
print("Assistant preview:", tokenizer.decode(sample_ids[sample_labels != -100]))

# Dataset wrapper
class PreTokenizedDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.data[idx]["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(self.data[idx]["attention_mask"],  dtype=torch.long),
            "labels":         torch.tensor(self.data[idx]["labels"],          dtype=torch.long),
        }

train_dataset = PreTokenizedDataset(tokenized_dataset)

# Train
training_args = TrainingArguments(
    per_device_train_batch_size   = 1,
    gradient_accumulation_steps   = 8,
    gradient_checkpointing        = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    warmup_ratio                  = 0.05,
    num_train_epochs              = 3,
    learning_rate                 = 1e-4,
    fp16                          = True,
    logging_steps                 = 5,
    optim                         = "paged_adamw_8bit",
    weight_decay                  = 0.01,
    lr_scheduler_type             = "cosine",
    seed                          = 3407,
    output_dir                    = drive_model_save_path,
    save_strategy                 = "epoch",
    save_total_limit              = 1,
    remove_unused_columns         = False,
    report_to                     = "none",
)

trainer = Trainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_dataset,
    data_collator = DataCollatorForSeq2Seq(
        tokenizer          = tokenizer,
        pad_to_multiple_of = 8,
        return_tensors     = "pt",
        padding            = True,
        label_pad_token_id = -100,
    ),
    args = training_args,
)

gc.collect()
torch.cuda.empty_cache()

print("\n--- Starting Training ---")
trainer_stats = trainer.train()

# Save locally
model.save_pretrained(drive_model_save_path)
tokenizer.save_pretrained(drive_model_save_path)
print(f"\nSaved to {drive_model_save_path}")

# Inference
!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 -q
!pip install bitsandbytes==0.43.3 -q
!pip install transformers==4.44.0 peft==0.12.0 accelerate==0.34.0 -q

import os
import gc
import torch
import warnings
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

model_name            = "Qwen/Qwen2.5-3B-Instruct"
drive_model_save_path = "/content/drive/My Drive/Personal Project/Legal_Contracts_Simplify/latest/legal_chatbot_qwen_lora_model"
MAX_SEQ_LENGTH        = 5120

# INITIALIZE THE BGE-M3 GUARDRAIL PIPELINE
print("Initializing Zero-Shot Guardrail Engine...")
guard_classifier = pipeline(
    "zero-shot-classification",
    model  = "MoritzLaurer/bge-m3-zeroshot-v2.0",
    device = 0 if torch.cuda.is_available() else -1
)

# LOAD GENERATIVE COMPONENT (QWEN + LORA)
print("Loading Generative LLM Layers...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Left-pad for generation

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map          = "auto",
    torch_dtype         = torch.float16,
)

model_trained = PeftModel.from_pretrained(base_model, drive_model_save_path)
model_trained.eval()

print("\n--- System Setup Complete: Model ready for inference ---\n")

system_prompt = """<persona>
You are a professional legal analyst. You simplify complex legal documents into clear, plain English for non-lawyers.
</persona>

<methodology>
Always follow exactly two phases. Never skip or merge them.

PHASE 1 — Output under the tag '<Reasoning>:' — Do these three steps:
  STEP 1 - IDENTIFY: State the document type, jurisdiction if mentioned, and its legal purpose.
  STEP 2 - TRANSLATE: List every technical legal term and define it in plain English.
  STEP 3 - DECONSTRUCT: Trace the logic step-by-step. Expose all conditions, exceptions, and jurisdiction-specific rules.

PHASE 2 — Output under the tag '<Summary>:' — Write a structured summary:
  - Write as a standalone document. Never copy or restate the Reasoning.
  - Use single markdown headers starting with exactly three hashes: ### Section Name
  - Every primary bullet MUST start with a plain asterisk and look exactly like this: * Topic Name: Description
  - Each primary bullet description must be exactly 1-2 overview sentences maximum.
  - If a topic contains an itemized, comma-separated, or distinct list of variables (such as multiple countries, data types, prohibitions, or conditions), you are strictly FORBIDDEN from writing them as a single continuous sentence. You MUST expand them as individual indented sub-bullets beneath that primary point:
      - Sub-item Name: Plain description of rule or item.
  - No item, legal exception, or parameter from the source text may be skipped, combined, or omitted.
</methodology>

<rules>
1. COMPLETENESS: Every single section, rule, data type, jurisdiction, and condition in the source text must appear explicitly. If a list contains 14 items, you must generate 14 distinct sub-bullets.
2. CONDITIONALS: Never turn a conditional into an absolute. If the source says "only in limited countries" or "depending on your settings", you must keep that exact condition in your output.
3. DIRECTIONALITY: Always state who owes what to whom. Never reverse company obligations and user obligations.
4. LIABILITY: Flag any "as-is", warranty disclaimers, or penalty assignments explicitly.
5. READING LEVEL: Write at a 6th-grade reading level throughout.
</rules>"""

# INFERENCE WRAPPER
def generate_simplified_legal_text(user_input_text):
    # 1. Clean raw whitespaces and formatting blocks
    user_input_clean = str(user_input_text).replace('\xa0', ' ')
    user_input_clean = " ".join(user_input_clean.split())

    # 2. PHASE A: High-Speed Zero-Shot Filtering
    candidate_labels = [
        "legal contract clause, agreement, terms of use, or privacy policy",
        "general conversation, question, code, or unrelated document"
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*The attention mask is not set.*")
        router_result = guard_classifier(user_input_clean, candidate_labels, truncation=True)

    legal_index = router_result['labels'].index("legal contract clause, agreement, terms of use, or privacy policy")
    legal_score = router_result['scores'][legal_index]

    print(f"DEBUG -> Guardrail Validation Score: {legal_score:.4f}")

    if legal_score < 0.60:
        return "I am not aware of this!!!"

    print("DEBUG -> Verification Passed. Invoking fine-tuned generative layer...")

    # 3. PHASE B: Fine-Tuned Generation Sequence
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"<contract_clause>\n{user_input_clean}\n</contract_clause>"
                "\n<task>\n"
                "Analyze and simplify the following legal text. Apply your full CoT methodology "
                "and produce a complete structured summary as specified in your instructions.\n"
                "</task>"
            ),
        },
    ]

    # Tokenize
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize             = True,
        add_generation_prompt = True,
        return_tensors       = "pt"
    )

    attention_mask = (input_ids != tokenizer.eos_token_id).to(torch.long)

    # Truncate if input exceeds MAX_SEQ_LENGTH
    if input_ids.shape[-1] > MAX_SEQ_LENGTH:
        input_ids      = input_ids[:, -MAX_SEQ_LENGTH:]
        attention_mask = attention_mask[:, -MAX_SEQ_LENGTH:]
        print(f"WARNING -> Input truncated to {MAX_SEQ_LENGTH} tokens.")

    # Generate
    with torch.no_grad():
        output = model_trained.generate(
            input_ids        = input_ids.to("cuda"),
            attention_mask   = attention_mask.to("cuda"),
            max_new_tokens   = 2048,
            do_sample        = False,
            no_repeat_ngram_size = 5,
            temperature      = None,
            top_p            = None,
            top_k            = None,
            pad_token_id     = tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

print("Inference Pipeline Ready!!!")

# Test Case 1: Valid Legal Document Context
valid_clause = "The user explicitly grants the application an assignment license to hold non-exclusive media distribution logs."
print("\n[Executing Test Case 1: Valid Legal Domain]")
print(generate_simplified_legal_text(valid_clause))

print("-" * 50)

# Test Case 2: Out of Domain (Adversarial/General Text)
invalid_query = "What is the best recipe to bake a low-calorie chocolate lava cake at home?"
print("\n[Executing Test Case 2: Out of Domain Domain]")
print(generate_simplified_legal_text(invalid_query))

# Sample Examples for testing

# 1. Test Out-of-Domain (Should print: "I am not aware of this")
print("Test 1 (Coding Prompt):")
print(generate_simplified_legal_text("Can you help me build a web scraper in Python?"))
print("-" * 40)

# 2. Test Smalltalk (Should print: "I am not aware of this")
print("Test 2 (Smalltalk Prompt):")
print(generate_simplified_legal_text("Hello! What are your thoughts on the weather today?"))
print("-" * 40)

# 3. Test Real Legalese (Should pass the guardrail and produce the output)
test_clause = (
    "The Customer agrees to indemnify, defend, and hold harmless the Provider from and against "
    "any and all claims, liabilities, damages, losses, or expenses arising out of or in any way "
    "connected with the Customer's misuse of the software platform."
)
print("Test 3 (Real Contract Clause):")
print(generate_simplified_legal_text(test_clause))

# Some Real Life Texts

text2= "who is the prime minister of India"
print(generate_simplified_legal_text(text2))

text3 = """Apple Website Terms of Use
Legal Information & Notices
Ownership of Site; Agreement to Terms of Use
These Terms and Conditions of Use (the "Terms of Use") apply to the Apple web site located at www.apple.com, and all associated sites linked to www.apple.com by Apple, its subsidiaries and affiliates, including Apple sites around the world (collectively, the "Site"). The Site is the property of Apple Inc. ("Apple") and its licensors. BY USING THE SITE, YOU AGREE TO THESE TERMS OF USE; IF YOU DO NOT AGREE, DO NOT USE THE SITE.
Apple reserves the right, at its sole discretion, to change, modify, add or remove portions of these Terms of Use, at any time. It is your responsibility to check these Terms of Use periodically for changes. Your continued use of the Site following the posting of changes will mean that you accept and agree to the changes. As long as you comply with these Terms of Use, Apple grants you a personal, non-exclusive, non-transferable, limited privilege to enter and use the Site.
Content
All text, graphics, user interfaces, visual interfaces, photographs, trademarks, logos, sounds, music, artwork and computer code (collectively, "Content"), including but not limited to the design, structure, selection, coordination, expression, "look and feel" and arrangement of such Content, contained on the Site is owned, controlled or licensed by or to Apple, and is protected by trade dress, copyright, patent and trademark laws, and various other intellectual property rights and unfair competition laws.
Except as expressly provided in these Terms of Use, no part of the Site and no Content may be copied, reproduced, republished, uploaded, posted, publicly displayed, encoded, translated, transmitted or distributed in any way (including "mirroring") to any other computer, server, Web site or other medium for publication or distribution or for any commercial enterprise, without Apple’s express prior written consent.
You may use information on Apple products and services (such as data sheets, knowledge base articles, and similar materials) purposely made available by Apple for downloading from the Site, provided that you (1) not remove any proprietary notice language in all copies of such documents, (2) use such information only for your personal, non-commercial informational purpose and do not copy or post such information on any networked computer or broadcast it in any media, (3) make no modifications to any such information, and (4) not make any additional representations or warranties relating to such documents.
Your Use of the Site
You may not use any "deep-link", "page-scrape", "robot", "spider" or other automatic device, program, algorithm or methodology, or any similar or equivalent manual process, to access, acquire, copy or monitor any portion of the Site or any Content, or in any way reproduce or circumvent the navigational structure or presentation of the Site or any Content, to obtain or attempt to obtain any materials, documents or information through any means not purposely made available through the Site. Apple reserves the right to bar any such activity.
You may not attempt to gain unauthorized access to any portion or feature of the Site, or any other systems or networks connected to the Site or to any Apple server, or to any of the services offered on or through the Site, by hacking, password "mining" or any other illegitimate means.
You may not probe, scan or test the vulnerability of the Site or any network connected to the Site, nor breach the security or authentication measures on the Site or any network connected to the Site. You may not reverse look-up, trace or seek to trace any information on any other user of or visitor to the Site, or any other customer of Apple, including any Apple account not owned by you, to its source, or exploit the Site or any service or information made available or offered by or through the Site, in any way where the purpose is to reveal any information, including but not limited to personal identification or information, other than your own information, as provided for by the Site.
You agree that you will not take any action that imposes an unreasonable or disproportionately large load on the infrastructure of the Site or Apple’s systems or networks, or any systems or networks connected to the Site or to Apple.
You agree not to use any device, software or routine to interfere or attempt to interfere with the proper working of the Site or any transaction being conducted on the Site, or with any other person’s use of the Site.
You may not forge headers or otherwise manipulate identifiers in order to disguise the origin of any message or transmittal you send to Apple on or through the Site or any service offered on or through the Site. You may not pretend that you are, or that you represent, someone else, or impersonate any other individual or entity.
You may not use the Site or any Content for any purpose that is unlawful or prohibited by these Terms of Use, or to solicit the performance of any illegal activity or other activity which infringes the rights of Apple or others.
Purchases; Other Terms and Conditions
Additional terms and conditions may apply to purchases of goods or services and to specific portions or features of the Site, including contests, promotions or other similar features, all of which terms are made a part of these Terms of Use by this reference. You agree to abide by such other terms and conditions, including where applicable representing that you are of sufficient legal age to use or participate in such service or feature. If there is a conflict between these Terms of Use and the terms posted for or applicable to a specific portion of the Site or for any service offered on or through the Site, the latter terms shall control with respect to your use of that portion of the Site or the specific service.
Apple’s obligations, if any, with regard to its products and services are governed solely by the agreements pursuant to which they are provided, and nothing on this Site should be construed to alter such agreements.
Apple may make changes to any products or services offered on the Site, or to the applicable prices for any such products or services, at any time, without notice. The materials on the Site with respect to products and services may be out of date, and Apple makes no commitment to update the materials on the Site with respect to such products and services.
The following terms also govern and apply to your use of the Site, and they are incorporated herein by this reference:
Guidelines for Using Apple Trademarks & Copyrights
Rights & Permissions
Trademarks
Claims of Copyright Infringement
Piracy
Counterfeit Products
Apple’s Unsolicited Idea Submission Policy
Software License Information
Legal Contacts
Each of these policies may be changed from time to time and are effective immediately upon posting such changes on the Site.
Accounts, Passwords and Security
Certain features or services offered on or through the Site may require you to open an account (including setting up an Apple ID and password). You are entirely responsible for maintaining the confidentiality of the information you hold for your account, including your password, and for any and all activity that occurs under your account as a result of your failing to keep this information secure and confidential. You agree to notify Apple immediately of any unauthorized use of your account or password, or any other breach of security. You may be held liable for losses incurred by Apple or any other user of or visitor to the Site due to someone else using your Apple ID, password or account as a result of your failing to keep your account information secure and confidential.
You may not use anyone else’s Apple ID, password or account at any time without the express permission and consent of the holder of that Apple ID, password or account. Apple cannot and will not be liable for any loss or damage arising from your failure to comply with these obligations.
Privacy
Apple’s Privacy Policy applies to use of this Site, and its terms are made a part of these Terms of Use by this reference. To view Apple’s Privacy Policy, click here. Additionally, by using the Site, you acknowledge and agree that Internet transmissions are never completely private or secure. You understand that any message or information you send to the Site may be read or intercepted by others, even if there is a special notice that a particular transmission (for example, credit card information) is encrypted.
Links to Other Sites and to the Apple Site
This Site may contain links to other independent third-party Web sites ("Linked Sites"). These Linked Sites are provided solely as a convenience to our visitors. Such Linked Sites are not under Apple’s control, and Apple is not responsible for and does not endorse the content of such Linked Sites, including any information or materials contained on such Linked Sites. You will need to make your own independent judgment regarding your interaction with these Linked Sites.
"""

print(generate_simplified_legal_text(text3))

text4 = """Service terms
These Service Terms govern your use of the Services. Capitalized terms not defined here will have the meanings in the Terms of Use, the OpenAI Services Agreement⁠⁠, or other agreement you have with us governing your use of the Services (“Agreement”). If there is a conflict between the Service Terms and your Agreement, the Service Terms will control. For purposes of these Terms, “Content” includes “Customer Content.”

1. API
OpenAI’s indemnification obligations to API customers under the Agreement include any third party claim that Customer’s use or distribution of Output infringes a third party’s intellectual property right. This indemnity does not apply where: (i) Customer or Customer’s End Users knew or should have known the Output was infringing or likely to infringe, (ii) Customer or Customer’s End Users disabled, ignored, or did not use any relevant citation, filtering or safety features or restrictions provided by OpenAI, (iii) Output was modified, transformed, or used in combination with products or services not provided by or on behalf of OpenAI, (iv) Customer or its End Users did not have the right to use the Input or fine-tuning files to generate the allegedly infringing Output, (v) the claim alleges violation of trademark or related rights based on Customer’s or its End Users’ use of Output in trade or commerce, and (vi) the allegedly infringing Output is from content from a Third Party Offering.

Customer will only, and will ensure that its End Users only, use APIs in accordance with the applicable documentation at https://platform.openai.com/docs⁠⁠⁠(opens in a new window).

2. Beta Services
This section governs your use of services or features that OpenAI offers on an alpha, preview, early access, or beta basis (“Beta Services”). Beta Services are offered “as-is” to allow testing and evaluation and are excluded from any indemnification obligations OpenAI may have to you.

OpenAI makes no representations or warranties for Beta Services, including any warranty that Beta Services will be generally available, uninterrupted or error-free, or that Content will be secure or not lost or damaged. Except to the extent prohibited by law, OpenAI expressly disclaims all warranties for Beta Services, including any implied warranties of merchantability, satisfactory quality, fitness for a particular purpose, non-infringement, or quiet enjoyment, and any warranties arising out of any course of dealing or usage of trade.

3. ChatGPT Enterprise, Edu, Healthcare and Business
(a) Administrators. ChatGPT Enterprise, ChatGPT Edu, and ChatGPT for Healthcare (collectively, “Enterprise”) and ChatGPT Business accounts are managed by End Users with administrative privileges (“Administrators”). Administrators may be able to add, remove and suspend End Users’ access to the organization’s workspace. In addition, ChatGPT Enterprise Administrators may be able to (a) access, share and remove Content; and (b) access logging and information about End Users’ use of ChatGPT Enterprise. Customers are responsible for obtaining and maintaining all necessary consents from End Users to take the actions above and to allow OpenAI to deliver the Services.

(b) Output indemnity. OpenAI’s indemnification obligations to Enterprise customers under the Agreement include claims that Customer’s use or distribution of Output infringes a third party’s intellectual property right. This indemnity does not apply where: (i) Customer or Customer’s End Users knew or should have known the Output was infringing or likely to infringe, (ii) Customer or Customer’s End Users disabled, ignored, or did not use any relevant citation, filtering or safety features or restrictions provided by OpenAI, (iii) Output was modified, transformed, or used in combination with products or services not provided by or on behalf of OpenAI, (iv) Customer or its End Users did not have the right to use the Input or fine-tuning files to generate the allegedly infringing Output, (v) the claim alleges violation of trademark or related rights based on Customer’s or its End Users’ use of Output in trade or commerce, and (vi) the allegedly infringing Output is from content from a Third Party Offering.

(c) ChatGPT for Healthcare. (i) You should always verify the information provided by ChatGPT for Healthcare and exercise independent professional judgment in decision-making about a patient without relying primarily or solely on the output. Do not use ChatGPT for Healthcare to analyze medical images or patterns/signals from signal acquisition systems or in vitro diagnostic devices (such as an ECG waveforms or genomic sequences). (ii) Customer’s Business Associate and Healthcare Addendum applies to Customer’s use of ChatGPT for Healthcare.

4. Codex and Code Generation
Output generated by code generation features of our Services, including OpenAI Codex⁠⁠⁠, may be subject to third party licenses, including, without limitation, open source licenses.

5. GPTs
Users can create and share access to their own customized versions of ChatGPT called “GPTs”.

For Builders of GPTs:
(a) GPT Content. The information or content that you upload to or include with your GPT (for example your GPT name, instructions, and description) (“GPT Content”) is your Content. As between you and OpenAI, you are solely responsible for your GPT Content, Actions, and configurations that you use or enable to create your GPT as well as any Output that is based on your GPT Content, Actions, and configurations. You must ensure your GPT complies with the Agreement and our Usage Policies⁠⁠⁠.

(b) Distribution and Promotion of GPTs. By sharing your GPT with others, you grant a nonexclusive, worldwide, irrevocable, royalty-free license: (i) to OpenAI to use, test, store, copy, translate, display, modify, distribute, promote, and otherwise make available to other users all or any part of your GPT (including GPT Content); and (ii) to the extent Output from your GPT includes your GPT Content, to users of your GPT to use, store, copy, display, distribute, prepare derivative works of and otherwise use your GPT Content. You will ensure that all information that you publish about your GPT is, at all times, complete, accurate, and not misleading.

(c) Actions. Any API, website, or service that interacts with a GPT (an “Action”) is subject to our App Developer Terms⁠⁠ and you are responsible for ensuring that any Action included with your GPT operates in compliance with those terms.

(d) Removal. We may reject or remove any GPT from our Services at any time for any reason without notice to you, such as for legal or security reasons or if your GPT violates our Terms.

For Users of GPTs:
(a) Third Party GPTs. Except where OpenAI is identified as the builder of a GPT, GPTs are created by other users and they may rely on content or third party applications that are not controlled by OpenAI. Use of “GPT” in the name of a GPT created by other users does not imply that OpenAI created, supports or endorses the GPT. Only use GPTs that you know and trust.

(b) Abuse Reporting. You can report GPTs that violate our Usage Policies⁠⁠⁠ using our reporting feature within ChatGPT.

(c) Actions. GPTs may allow you to interact with Actions. Those Actions are subject to our terms for Apps and Actions below.

(d) Changes and Removal. OpenAI and creators of GPTs can remove GPTs at any time for any reason without prior notice.

6. Image and Video Capabilities
Our models can accept images and videos as part of Inputs to the Services (“Visual Capabilities”). You may not use Visual Capabilities to assist in identifying a person nor to solicit or infer private or sensitive information about a person. You may not use Visual Capabilities to reproduce the likeness of any person without express consent and all necessary rights. By choosing to share an image or video publicly on the Services, such as by sharing it to the Sora feed or uploading it to create a Sora cameo, you represent you have all necessary rights, and you give OpenAI the right to reproduce, distribute, modify, display and perform it for the purpose of operating and promoting the Services. You also agree to give users the limited right to reproduce and remix that content solely on the Services. Uploading a cameo or sharing an image or video does not give other users any rights to use those materials outside of the Services."""

print(generate_simplified_legal_text(text4))

text6 = """
Our Usage Policy (also referred to as our “Acceptable Use Policy” or “AUP”) applies to anyone who can submit inputs to Anthropic’s products and/or services, including via any authorized resellers or passthrough access, all of whom we refer to as “users.” The Usage Policy is intended to help our users stay safe and promote the responsible use of our products and services.

The Usage Policy is categorized according to who can use our products and for what purposes. We will update our policy as our technology and the associated risks evolve or as we learn about unanticipated risks.

Universal Usage Standards: Our Universal Usage Standards apply to all users and use cases.
High-Risk Use Case Requirements: Our High-Risk Use Case Requirements apply to specific consumer-facing use cases that pose an elevated risk of harm.
Additional Use Case Guidelines: Our Additional Use Case Guidelines apply to certain other use cases, including consumer-facing chatbots, products serving minors, agentic use, and Model Context Protocol servers.
Anthropic’s Safeguards Team will implement detection and monitoring to enforce our Usage Policy, so please review this policy carefully before using our products or services. If we learn that you have violated our Usage Policy, we may throttle, suspend, or terminate your access to our products and services. We may also block or modify model outputs when inputs violate our Usage Policy.

If you believe that our model outputs are potentially inaccurate, biased or harmful, please notify us at usersafety@anthropic.com, or report it directly in our product through the “report issues” thumbs down button or similar feedback features (where available). You can read more about our Safeguards practices and recommendations in our Safeguards Support Center.

This Usage Policy is calibrated to strike an optimal balance between enabling beneficial uses and mitigating potential harms. Anthropic may enter into contracts with certain governmental customers that tailor use restrictions to that customer’s public mission and legal authorities if, in Anthropic’s judgment, the contractual use restrictions and applicable safeguards are adequate to mitigate the potential harms addressed by this Usage Policy.

Universal Usage Standards


Do Not Violate Applicable Laws or Engage in Illegal Activity



Do Not Compromise Critical Infrastructure



Do Not Compromise Computer or Network Systems



Do Not Develop or Design Weapons



Do Not Incite Violence or Hateful Behavior



Do Not Compromise Privacy or Identity Rights



Do Not Compromise Children’s Safety




Do Not Create Psychologically or Emotionally Harmful Content



Do Not Create or Spread Misinformation



Do Not Undermine Democratic Processes or Engage in Targeted Campaign Activities



Do Not Use for Criminal Justice, Censorship, Surveillance, or Prohibited Law Enforcement Purposes



Do Not Engage in Fraudulent, Abusive, or Predatory Practices



Do Not Abuse our Platform



Do Not Generate Sexually Explicit Content


High-Risk Use Case Requirements
Some use cases pose an elevated risk of harm because they influence domains that are vital to public welfare and social equity. For these use cases, given potential risks to individuals and consumers, we believe that relevant human expertise should be integrated and that end-users should be aware when AI has been involved in producing outputs.

As such, for the “High-Risk Use Cases” described below, we require that you implement these additional safety measures:

Human-in-the-loop: When using our products or services to provide advice, recommendations, or in subjective decision-making directly affecting individuals or consumers, a qualified professional in that field must review the content or decision prior to dissemination or finalization. You or your organization are responsible for the accuracy and appropriateness of that information.
Disclosure: If model outputs are presented directly to individuals or consumers, you must disclose to them that you are using AI to help produce your advice, decisions, or recommendations. This disclosure must be provided at a minimum at the beginning of each session.
“High-Risk Use Cases” include:

Legal: Use cases related to legal interpretation, legal guidance, or decisions with legal implications
Healthcare: Use cases related to healthcare decisions, medical diagnosis, patient care, therapy, mental health, or other medical guidance. Wellness advice (e.g., advice on sleep, stress, nutrition, exercise, etc.) does not fall under this category
Insurance: Use cases related to health, life, property, disability, or other types of insurance underwriting, claims processing, or coverage decisions
Finance: Use cases related to financial decisions, including investment advice, loan approvals, and determining financial eligibility or creditworthiness
Employment and housing: Use cases related to decisions about the employability of individuals, resume screening, hiring tools, or other employment determinations or decisions regarding eligibility for housing, including leases and home loans
Academic testing, accreditation and admissions: Use cases related to standardized testing companies that administer school admissions (including evaluating, scoring or ranking prospective students), language proficiency, or professional certification exams; agencies that evaluate and certify educational institutions
Media or professional journalistic content: Use cases related to using our products or services to automatically generate content and publish it for external consumption
Additional Use Case Guidelines
The below use cases – regardless of whether they are High-Risk Use Cases – must comply with the additional guidance provided.

All consumer-facing chatbots, including any external-facing or interactive AI agent, must disclose to users that they are interacting with AI rather than a human. This disclosure must be provided at a minimum at the beginning of each chat session.
Products serving minors, including organizations providing minors with the ability to directly interact with products that incorporate our API(s), must comply with the additional guidelines outlined in our Help Center article.
Agentic use cases must still comply with the Usage Policy. We provide examples of Usage Policy prohibitions in the context of agentic use in this Help Center article.
Model Context Protocol (MCP) servers listed in our Connector Directory must comply with our Directory Policy."""

print(generate_simplified_legal_text(text6))


