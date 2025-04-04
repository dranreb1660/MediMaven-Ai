# pipelines/llama2_finetune_pipeline.py

import os, time,math
import wandb
import torch
import pandas as pd
from typing import Dict, Any, List, Tuple, Type, Union

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from zenml.steps import step
from zenml.pipelines import pipeline

from src.utils import chunk_text_by_tokens, get_mongo_connection
from zenml.materializers.base_materializer import BaseMaterializer
from datasets import Dataset
from tqdm.auto import tqdm

# -----------------------------------------------------------------
#  Database connection (not changed)
# -----------------------------------------------------------------
db = get_mongo_connection()

# --------------------------------------------------------------
#   STEP 1: Fetch Data
# --------------------------------------------------------------
@step
def fetch_finetune_data(
    mongodb_collection: str = "qa_master_processed",
) -> pd.DataFrame:
    """
    Fetch data from MongoDB (or any other storage).
    Each record has: [question, context, answer, context_id, etc.].
    """
    collection = db[mongodb_collection]
    data = list(collection.find({}, {'_id': 0}))
    df = pd.DataFrame(data)
    if "context_id" not in df.columns:
        df["context_id"] = df.index.astype(str)
    return df

# --------------------------------------------------------------
#   STEP 2: Chunk Large Contexts
# --------------------------------------------------------------
@step
def chunk_llm_docs(df: pd.DataFrame, base_model_name, tokenizer=None) -> pd.DataFrame:
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    all_rows = []
    for _, row in df.iterrows():
        original_text = row["context"]
        # Overlap chunking: 512 max tokens, 256 stride
        chunked_texts = chunk_text_by_tokens(original_text, tokenizer,
                                             max_tokens=512, overlap=50)
        for idx, chunk in enumerate(chunked_texts):
            new_row = row.copy()
            new_row["context"] = chunk
            new_row["chunk_id"] = f"{row['context_id']}_{idx}"
            all_rows.append(new_row)

    elapsed = time.time() - start_time
    print(f'Chunking took {elapsed:.2f} sec, produced {len(all_rows)} rows.')

    wandb.init(project="MediMaven-LLM-Finetuning",
               job_type="llm_finetune_pipeline",
               reinit=True)
    wandb.log({
        "num_original_docs": df.shape[0],
        "num_new_docs": len(all_rows),
        "chunking_time_sec": elapsed
    })

    return pd.DataFrame(all_rows)

# --------------------------------------------------------------
#   STEP 3: Prepare Instruction Prompt
# --------------------------------------------------------------
def build_prompt(row):
    sys_msg = (
        "You are a helpful medical AI assistant. "
        "Provide clear and concise answers based on the given context."
    )
    question = row["question"]
    context = row["context"]
    # Example format for Llama instructions:
    prompt = (
        f"[INST] <<SYS>>\n{sys_msg}\n<<SYS>>\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        "[/INST]"
    )
    return prompt

@step
def prepare_instruction_text(df: pd.DataFrame) -> pd.DataFrame:
    df["instruction_text"] = df.apply(build_prompt, axis=1)
    return df

# --------------------------------------------------------------
#   STEP 4: Tokenize Data
# --------------------------------------------------------------
@step
def tokenize_data(
    df: pd.DataFrame,
    max_length: int,
    base_model_name: str,
    tokenizer = None
) -> Dataset:
    """
    Tokenizes the instruction text + answer for Causal LM.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  
        
    def build_full_text(example):
        return example["instruction_text"] + " " + example["answer"]

    data_dict = {
        "text": df.apply(build_full_text, axis=1).tolist(),
        "instruction_text": df["instruction_text"].tolist(),
        "answer": df["answer"].tolist()
    }

    try:
        ds = Dataset.from_dict(data_dict)
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

        ds = ds.map(tokenize_function, batched=True)
        ds = ds.remove_columns(["text", "instruction_text", "answer"])
        
        # Verify the dataset has the expected columns
        if "input_ids" not in ds.features:
            raise ValueError("Tokenization failed - input_ids not found in dataset")
        
        print(f"Tokenization successful. Dataset size: {len(ds)}")
        print(ds[0])
        return ds
    except Exception as e:
        raise ValueError(f"Failed to tokenize data: {str(e)}")

# -------------------------------------------------------------------
#  Step 5: Three way split data
# -------------------------------------------------------------------
# Update the split_dataset step to add validation
@step
def split_dataset(ds: Dataset) -> Dict[str, Dataset]:
    if ds is None:
        raise ValueError("Input dataset is None")
    if len(ds) == 0:
        raise ValueError("Input dataset is empty")
        
    try:
        # First split: 10% test, 90% train
        d1 = ds.train_test_split(test_size=0.1, seed=42)
        # Second split: 10% of remaining 90% (9% total) for validation, 81% for train
        d2 = d1["train"].train_test_split(test_size=0.1, seed=42)
        
        print(f"Split successful. Dataset sizes:")
        print(f'train: {len(d2["train"])}\n'
            f'validation: {len(d2["test"])}\n'
            f'test: {len(d1["test"])}')
        return {
            "train": d2["train"],
            "validation": d2["test"],
            "test": d1["test"]
        }
    except Exception as e:
        raise ValueError(f"Failed to split dataset: {str(e)}")

# -------------------------------------------------------------------
#  Utility: Create Model & Tokenizer
# -------------------------------------------------------------------
def create_model_and_tokenizer(MODEL_NAME: str):
    """
    Loads an LLM in 4-bit mode for QLoRA. Adjust as needed.
    """
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# --------------------------------------------------------------
#   STEP 5: Train Model with Basic Param Tuning
# --------------------------------------------------------------
@step(enable_cache=True)
def train_model(
    base_model_name: str,
    split_ds: Dict[str, Any],
    max_len: int,
    output_dir: str = "./models/llama_finetuned",
    # We will do a mini param search:
    lr: float = 5e-5,
    lora_r: int= 16,
    epochs: int = 2,
) -> str:
    
    """
    Fine-tunes the Llama model with QLoRA, doing a simple loop over
    different (lr, lora_r) combos. Logs each run to W&B, picks the best.

    Returns the best model's final path.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

    model, tokenizer = create_model_and_tokenizer(base_model_name)

    train_ds = split_ds["train"]
    valid_ds = split_ds["validation"]

    response_template = "[/INST]"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )



    run_name = f"lr={lr}-r={lora_r}"

    # Re-init a fresh LoRA adapter for each run
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            # "mlp.gate_proj",
            # "mlp.up_proj",
            # "mlp.down_proj", 
            ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_lora = get_peft_model(model, lora_config)

    # SFT Trainer config
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=max_len,
        per_device_train_batch_size=14,
        per_device_eval_batch_size=14,
        gradient_accumulation_steps=6,
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=lr,
        bf16=True,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="wandb",
        save_safetensors=True,
        dataset_kwargs={"add_special_tokens": False},
        seed=42
    )

    wandb.init(resume="auto",
        # project="MediMaven-LLM-Finetuning",
        # job_type="finetune",
        # name=f"llama_finetune_{run_name}",
    )
    trainer = SFTTrainer(
        model=model_lora,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=sft_config,
        peft_config=lora_config,
        data_collator=data_collator,
    )

    trainer.train()

    # Evaluate perplexity on validation set
    eval_metrics = trainer.evaluate()
    # "eval_loss" is typically the metric returned
    val_loss = eval_metrics.get("eval_loss", 999.0)
    val_ppl = float(torch.exp(torch.tensor(val_loss)))

    wandb.log({"val_ppl": val_ppl})
    wandb.finish()

    # If best so far, save
    trainer.save_model(output_dir)

    print(f"Model path: {output_dir} => PPL={val_ppl:.3f}")
    return output_dir

# --------------------------------------------------------------
#   STEP 6: Merge & Evaluate on Held-Out Test
# --------------------------------------------------------------
import torch
# import math
from tqdm.auto import tqdm

def parse_prompt_and_answer(text: str) -> (str, str):
    """
    Splits the text at '[/INST]' to separate the prompt portion from the answer.
    Example input:
        "[INST] question context [/INST] some answer text"
    Returns:
        prompt_str: "[INST] question context [/INST]"
        answer_str: "some answer text"
    """
    marker = "[/INST]"
    idx = text.find(marker)
    if idx == -1:
        # If there's no [/INST], treat the entire text as prompt (edge case).
        return text, ""
    prompt_str = text[: idx + len(marker)]
    answer_str = text[idx + len(marker) :]
    return prompt_str.strip(), answer_str.strip()

def evaluate_in_batches(
    model,
    dataset,
    tokenizer,
    batch_size=8,
    max_new_tokens=300,
    device="cuda"
):
    """
    Evaluate a model's "answer perplexity" on a large dataset where each example
    contains "[INST] prompt [/INST] answer" in its 'input_ids'.

    *Batched approach*:
      1) We decode to find prompt vs. answer, but feed only the *prompt* to
         `model.generate()` (no answer leak).
      2) We compute perplexity on the *answer* portion only by masking prompt tokens.

    Returns:
        ppl: A float perplexity measuring how well the model predicts the answer.
        logs: A list of example logs with {prompt, prediction, ground_truth, loss}.
    """
    model.eval()
    model.to(device)
    
    logs = []
    total_loss = 0.0
    total_samples = 0

    # Iterate in mini-batches
    for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        end_idx = start_idx + batch_size
        # NOTE: Slicing a HF Dataset returns a dict of columns, each a list
        batch_slice = dataset[start_idx:end_idx]

        # Extract the 'input_ids' list (size: <= batch_size)
        input_ids_batch = batch_slice["input_ids"]

        # --- 1) For each sample, decode, split into prompt vs. answer. ---
        prompts, answers = [], []
        for ids in input_ids_batch:
            full_text = tokenizer.decode(ids, skip_special_tokens=False)
            full_text = full_text.replace("<|eot_id|>", "")
            prompt_str, answer_str = parse_prompt_and_answer(full_text)
            prompts.append(prompt_str)
            answers.append(answer_str)

        # --- 2) Generate from prompt-only ---
        prompt_enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            pred_ids = model.generate(
                input_ids=prompt_enc["input_ids"],
                attention_mask=prompt_enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        predictions = [
            tokenizer.decode(g, skip_special_tokens=True)
            for g in pred_ids
        ]
        predictions = [
            parse_prompt_and_answer(g)[1]  # extract answer portion
            for g in predictions
        ]

        # --- 3) Compute answer-only perplexity by masking prompt tokens ---
        # Re-tokenize full text: prompt + answer
        full_enc = tokenizer(
            [p + " " + a for p, a in zip(prompts, answers)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        # Also tokenize *just* the prompt to find how many tokens to mask
        prompt_only_enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        labels = full_enc["input_ids"].clone()
        for i_row in range(labels.size(0)):
            # number of tokens in the prompt
            prompt_len = prompt_only_enc["attention_mask"][i_row].sum().item()
            # mask out the prompt portion
            labels[i_row, :int(prompt_len)] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=full_enc["input_ids"],
                attention_mask=full_enc["attention_mask"],
                labels=labels
            )
        batch_loss = outputs.loss.item()  # average over all unmasked tokens in the batch

        # Weighted accumulation: multiply avg loss by number of samples
        num_in_batch = len(prompts)
        total_loss += batch_loss * num_in_batch
        total_samples += num_in_batch

        # --- 4) Log a few examples for debugging ---
        # We'll log 2 examples each batch or so
        if len(logs) < 2 * ((start_idx // batch_size) + 1):
            for i_log in range(min(2, num_in_batch)):
                logs.append({
                    "prompt": prompts[i_log],
                    "prediction": predictions[i_log],
                    "ground_truth": answers[i_log],
                    "loss": batch_loss
                })

        # 7. Memory cleanup
        del full_enc, labels, predictions, outputs
        torch.cuda.empty_cache()


    # --- 5) Final perplexity across dataset (answer tokens only) ---
    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return ppl, logs

@step
def merge_and_evaluate(base_model_name: str, heldout_ds: Dict[str, Any], best_model_path: str = "models/llama_finetuned") -> float:
    """End-to-end merge, evaluate, and save in 4-bit"""
    from transformers import BitsAndBytesConfig
    import torch

    # Initialize W&B
    wandb.init(project="MediMaven-LLM-Finetuning", 
               job_type="merge_eval",
               reinit=True)

    # 1. Configure 4-bit from the start
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head"]  # Keep output layer precise
    )

    # 2. Load base model in 4-bit directly
    print("\nLoading base model in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # 3. Load LoRA adapter
    print("Loading LoRA adapter...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        best_model_path,
        device_map="auto"
    )

    # 4. Merge weights (automatically stays 4-bit)
    print("Merging LoRA weights...")
    merged_model = lora_model.merge_and_unload(
        progressbar=True,
        safe_merge=True
    )
    
    # Load tokenizer
    _, tokenizer = create_model_and_tokenizer(base_model_name)
    
    # 5. Evaluate with memory-safe batch size
    print("Running batched evaluation...")

    heldout_subset = heldout_ds["test"].shuffle(seed=42).select(range(1000))
    ppl, example_logs = evaluate_in_batches(
        model=merged_model,
        dataset=heldout_subset,
        tokenizer=tokenizer,
        batch_size=26 # Lower batch size for memory safety
    )
    
    # 6. Save models (4-bit config is preserved)
    print("Saving final 4-bit model...")
    merged_model.save_pretrained(
        "./models/merged_4bit",
        safe_serialization=True,
        max_shard_size="2GB"
    )
    
    # 7. Optional: Save FP16 version for reference
    # merged_model.to(torch.float16).save_pretrained("./models/merged_fp16")
    
    wandb.log({
        "perplexity": ppl,
        "examples": wandb.Table(dataframe=pd.DataFrame(example_logs))
    })
    
    return ppl
# --------------------------------------------------------------
#   PIPELINE DEFINITION
# --------------------------------------------------------------
@pipeline(enable_cache=True)  # Disable caching temporarily for debugging
def llama3_finetuning_pipeline():
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "./models/llama_finetuned"
    max_len = 768

    print("\n=== Starting Pipeline ===")
    
    # 1) Fetch
    df = fetch_finetune_data()
    
    # 2) Chunk
    df_chunked = chunk_llm_docs(df, MODEL_NAME)
    
    # 3) Prompt
    df_instructions = prepare_instruction_text(df_chunked)
    
    # 4) Tokenize
    tokenized_ds = tokenize_data(df_instructions, MODEL_NAME, max_length=max_len)
    
    # 5) Split
    split_ds = split_dataset(tokenized_ds)

    # 6) Train
    # best_model_dir = train_model(
    #     base_model_name=MODEL_NAME,
    #     split_ds=split_ds,
    #     max_len=max_len,
    #     output_dir=output_dir,
    #     lr=5e-5,
    #     lora_r=32,
    #     epochs=2
    # )
    # best_model_dir = "models/llama_finetuned"
    
    # 7) Merge & Evaluate
    merge_and_evaluate(
        # best_model_path=best_model_dir,
        base_model_name=MODEL_NAME,
        heldout_ds=split_ds
    )
    
    print("\n=== Pipeline Completed ===")




     
