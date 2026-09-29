# Legal-Clause-Simplification-using-Knowledge-Distillation

Transforming complex legal clauses into plain English explanations using Knowledge Distillation, Chain of Thought (CoT) Reasoning, QLoRA Fine-Tuning, and Lightweight CPU Deployment.

## Live Demo & Resources

| Resource | Description | Link |
|-----------|-------------|------|
| Web Application | Interactive legal clause simplification demo | [Launch App](https://ayushhh1662309-legal-snippet-simplifier.hf.space) |
| Fine-Tuned Model | Merged Qwen2.5-3B model after LoRA fine-tuning | [qwen2.5-3b-legal-merged](https://huggingface.co/ayushhh1662309/qwen2.5-3b-legal-merged) |
| Quantized Model | 5-bit GGUF model optimized for CPU inference | [qwen2.5-3b-legal-merged-Q5_K_M-GGUF](https://huggingface.co/ayushhh1662309/qwen2.5-3b-legal-merged-Q5_K_M-GGUF) |
| Space Repository | Hugging Face Space source code and deployment files | [View Files](https://huggingface.co/spaces/ayushhh1662309/Legal-Snippet-Simplifier/tree/main) |

---

## Overview

Legal contracts and agreements often contain dense language, legal jargon, and complex sentence structures that are difficult for non-lawyers to understand.

This project develops a specialized Legal Clause Simplification system that converts legal text into clear, human-readable explanations while preserving the original legal meaning.

The system follows a teacher-student training paradigm, where a larger language model generates reasoning traces and simplified explanations that are subsequently distilled into a smaller student model.

The final model was:

* Fine-tuned using LoRA + QLoRA.
* Quantized into GGUF format.
* Converted to 5-bit precision.
* Deployed on Hugging Face Spaces.
* Served entirely on the free CPU tier.
* Integrated with a Gradio-based interface that provides both simplified explanations and optional reasoning traces.

---

# Key Highlights

* Built a complete end-to-end legal NLP pipeline.
* Generated Chain-of-Thought reasoning traces using a teacher model through the Groq API.
* Created a distilled dataset containing complex legal clause, CoT reasoning and plain-English explanations.
* Fine-tuned Qwen2.5-3B-Instruct using LoRA and QLoRA.
* Computed training loss only on assistant responses while masking user prompt tokens.
* Added domain guardrails using zero-shot classification.
* Converted the model to 5-bit GGUF format for efficient inference.
* Deployed successfully on Hugging Face Spaces CPU infrastructure.
* Developed a Gradio-based web application with plain-English clause simplification and optional Chain-of-Thought reasoning visualization.

---

# Problem Statement

Legal clauses frequently contain:

* Complex legal terminology.
* Long and nested sentence structures.
* Ambiguous wording for non-experts.
* Hidden obligations and liabilities.

The goal of this project was to create a domain-specialized model capable of generating understandable explanations while preserving contractual intent and legal semantics.

---

# Project Pipeline

The project was developed through the following stages:

1. Collected multiple legal simplification datasets from Hugging Face.
2. Cleaned and standardized the datasets amd merged into a unified .csv file.
3. Generated reasoning traces using a teacher model accessed through the Groq API.
4. Constructed a Chain-of-Thought distilled dataset.
5. Fine-tuned a student model using QLoRA and LoRA.
6. Added inference-time domain guardrails using zero-shot classification.
7. Merged LoRA adapters into the base model.
8. Converted the model into GGUF format.
9. Quantized the model to 5-bit precision.
10. Deployed the application on Hugging Face Spaces using the free CPU tier.

---

# Dataset Preparation

The following datasets were collected and merged into a single training corpus.

* mteb/legal_summarization
* CodeHima/TOS_Dataset
* mteb/UnfairTOSLegalBenchClassification

Since each dataset followed a different schema, all datasets were transformed into a unified structure:

```json
{
  "legal_clause": "...",
  "plain_english": "..."
}
```

Preprocessing included:

* Column standardization.
* Duplicate removal.
* Null-value filtering.
* Formatting cleanup.
* Length validation.

This produced a consistent dataset suitable for knowledge distillation and fine-tuning.

---

# Knowledge Distillation

## Teacher Model

Llama-4-scout-17b-16e-instruct (17B active parameter) was accessed through the Groq API (serverless inference api) and used as the teacher model.

Instead of directly learning:

```text
Legal Clause → Simplified Explanation
```

the student model was trained on:

```text
Legal Clause → Reasoning Process → Simplified Explanation
```

This approach encourages the model to learn the underlying legal reasoning before generating simplified outputs.

Note: This teacher model was learned on 450 samples from the unified preprocessed dataset (csv file).

---

## Teacher Output Structure

Each legal clause was transformed into a structured reasoning format consisting of three stages:

```text
IDENTIFY
TRANSLATE
COGNITIVE TRACE
```

### IDENTIFY

Recognizes:

* document type
* jurisdiction (if mentioned)
* primary legal objective

### TRANSLATE

Converts legal terminology into plain language.

### COGNITIVE TRACE

The step-by-step logical breakdown of the text: who owns what, conditions and exceptions under which the rules are applicable, etc.

Example:

```text
Input:

The Licensee shall indemnify and hold harmless the Licensor.

Output:

IDENTIFY:
The licensee assumes liability.

TRANSLATE:
The licensee must cover losses.

COGNITIVE TRACE:
If legal issues arise because of the licensee's actions,
the licensee is responsible for handling those issues and costs.
```

---

## Distilled Dataset Construction

Teacher-generated outputs were combined with the original legal clauses to create the final distilled dataset.

Structure:

```json
{
  "input": "Legal Clause",
  "output": "Reasoning + Simplified Explanation"
}
```

Additional cleaning was performed to remove:

* Incomplete generations.
* Formatting inconsistencies.
* Duplicate samples.
* Invalid reasoning traces.

---

# Fine-Tuning

## Student Model

The student model used for fine-tuning was:

```text
Qwen2.5-3B-Instruct (3B parameters)
```

Training was performed using Google Colab on an NVIDIA Tesla T4 GPU.

---

## Training Environment

### Hardware

* Google Colab
* NVIDIA Tesla T4 GPU (16 GB VRAM)

### Core Libraries (versions)

```bash
torch==2.4.0
torchvision==0.19.0
bitsandbytes==0.43.3
transformers==4.44.0
peft==0.12.0
accelerate==0.34.0
```

---

## QLoRA Configuration

The model was loaded using 4-bit NF4 quantization through BitsAndBytes.

---

## LoRA Configuration

Adapters were applied to both attention and feed-forward layers:

```python
[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
```

Training parameters:

| Parameter     | Value     |
| ------------- | --------- |
| LoRA Rank (r) | 16        |
| LoRA Alpha    | 32        |
| LoRA Dropout  | 0.05      |
| Bias          | none      |
| Task Type     | CAUSAL_LM |

---

## Chat Template

The distilled dataset was converted into a conversational instruction-tuning format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Legal analyst instructions..."
    },
    {
      "role": "user",
      "content": "<task>...</task><contract_clause>...</contract_clause>"
    },
    {
      "role": "assistant",
      "content": "<Reasoning>...</Reasoning><Summary>...</Summary>"
    }
  ]
}
```

The dataset used explicit tags to separate:

- Task instructions
- Input legal clauses
- Reasoning traces
- Final simplified summaries

This structure enabled the model to learn a consistent reasoning pipeline before generating the final plain-English explanation.

---

## Training Strategy

The model was trained using Hugging Face Transformers and PEFT (Parameter Efficient Fine Tuning).

| Parameter             | Value            |
| --------------------- | ---------------- |
| Epochs                | 3                |
| Learning Rate         | 1e-4             |
| Batch Size            | 1                |
| Gradient Accumulation | 8                |
| Effective Batch Size  | 8                |
| Optimizer             | paged_adamw_8bit |
| Scheduler             | cosine           |
| Weight Decay          | 0.01             |
| Precision             | FP16             |

Note: To improve instruction-following behavior, loss was computed only on assistant responses while user prompt tokens were masked during training. This ensures the model learns to generate reasoning traces and simplified explanations rather than reproducing the input prompt.

---

# Domain Guardrails

To ensure the system remains focused on legal simplification tasks, a zero-shot classification layer was integrated before inference.

Model used:

```text
MoritzLaurer/bge-m3-zeroshot-v2.0
```

The classifier operates using Natural Language Inference (NLI).

Instead of directly predicting a category, the model evaluates whether a hypothesis statement is true or false.

Example:

```text
Input:
"The agreement shall remain valid for two years."

Hypothesis:
"This text is about legal contracts."

Result:
Entailment → Legal Query
```

Only queries classified as legal are forwarded to the fine-tuned student model, improving reliability and reducing off-domain responses.

---

# Inference Pipeline

During inference, the system performs the following steps:

1. Accepts a user-provided clause.
2. Runs zero-shot legal-domain classification.
3. Rejects non-legal inputs.
4. Sends valid legal clauses to the fine-tuned model.
5. Generates a plain-English explanation along with the CoT reasoning trace.
6. Returns the response to the user.

---

# Quantization and Deployment

After training:

1. LoRA adapters were merged with the base model.
2. The merged model was exported to Hugging Face.
3. The model was converted to GGUF format using [ggml-org/gguf-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo).
4. A 5-bit quantized version was generated.
5. The application was deployed on Hugging Face Spaces.

Benefits of GGUF deployment:

* Smaller model size.
* Lower RAM requirements.
* Faster CPU inference.
* Easier deployment.

Note: The application runs entirely on the Hugging Face Spaces free CPU tier, as a result the output generation can be slower for large clauses.

---

# Gradio Frontend

A lightweight Gradio interface was developed and integrated into Hugging Face Spaces to provide an interactive user experience.

### User Workflow

1. Enter a legal clause in the input text box.
2. The system validates whether the input belongs to the legal domain using zero-shot classification.
3. If the input is classified as legal, it is forwarded to the fine-tuned model.
4. The model generates:
   * A simplified plain-English explanation.
   * An optional reasoning trace **Show Reasoning Chain** showing how the conclusion was reached.
5. The results are displayed in separate output sections.

---

# Tech Stack

### Model Development

* Qwen2.5-3B-Instruct
* Hugging Face Transformers
* PEFT
* Accelerate
* BitsAndBytes

### Knowledge Distillation

* Llama-4-scout-17b-16e-instruct (via Groq API)
* Chain-of-Thought Distillation

### Training

* LoRA
* QLoRA
* Google Colab (T4 GPU)

### Deployment

* Converted to GGUF (5-bit precision)
* Hugging Face Hub
* Hugging Face Spaces

### Frontend

* Gradio

---

# Example

## Input

```text
The Customer agrees to indemnify, defend, and hold harmless the Provider from and against any and all claims, liabilities, damages, losses, or expenses arising out of or in any way connected with the Customer's misuse of the software platform.
```

## Output

The output is divided into two sections.

### Plain-English Summary

```text
* Indemnification Obligation: The Customer agrees to protect and shield the Provider from claims made against them.
* Scope of Protection: This protection covers all claims related to the misuse of the software platform by the Customer.
* Financial Responsibility: The Customer must cover any financial losses, liabilities, damages, or expenses incurred by the Provider due to the Customer's actions.
```

### CoT Reasoning

```text
1. IDENTIFY: This is a contractual indemnification clause, and its primary legal objective is to protect the Provider from claims made by third parties due to the Customer's misuse of the software platform.
2. TRANSLATE: The dense or ambiguous terms in this text include "indemnify," which means to compensate someone for losses; "hold harmless," which means to ensure that no one can sue you; and "claims, liabilities, damages, losses, or expenses," which refer to any financial or legal consequences.
3. COGNITIVE TRACE: 
   - Step 1: Identify the parties involved - The Customer agrees to indemnify, defend, and hold harmless the Provider.
   - Step 2: Understand the scope of protection - This includes claims arising out of or connected with the Customer's misuse of the software platform.
   - Step 3: Determine the obligations - The Customer must compensate for any losses, liabilities, damages, or expenses incurred by the Provider due to the Customer's actions.
```

---

# Results

The project successfully demonstrates:

* Chain-of-Thought Knowledge Distillation.
* Legal-domain reasoning supervision.
* Parameter-efficient fine-tuning (PEFT) using QLoRA.
* Assistant-only loss masking.
* Zero-shot clasiifcation: domain guardrails.
* Efficient GGUF quantization.
* CPU-only deployment on Hugging Face Spaces.
* Practical legal text simplification for real-world use cases.

---

# Repository Structure

The respository structure is:

```text
Legal-Clause-Simplification-using-Knowledge-Distillation/
│
├── Dataset/
│   ├── distilled_legal_dataset.csv
│   ├── distilled_state_tracker.csv
│   ├── master_compiled_legal_dataset.csv
|   ├── sft_chatbot_dataset.jsonl
|   ├── README.md
├── Scripts/
│   ├── Complete_Knowledge_Distillation_Notebook.py
│   ├── Upload_Unquantized_Model_to_HuggingFace.py
│   ├── README.md
└── README.md
```

---

# Disclaimer

This project is intended for educational and research purposes.

The generated outputs are simplified interpretations of legal text and should not be considered legal advice. Users should consult qualified legal professionals before making legal decisions based on generated content.

---

# Author

### Ayush Ghatak

• Generative AI • NLP • Legal AI • Large Language Models (LLMs) • Knowledge Distillation • Chain-of-Thought Distillation • Supervised Fine-Tuning • LoRA • QLoRA • PEFT • Prompt Engineering • Instruction Tuning • Dataset Engineering • Zero-Shot Classification • Natural Language Inference (NLI) • Model Quantization • GGUF • Hugging Face • Gradio • LLM Deployment 
