# pipelines/llama2_finetune_pipeline.py

import os, time
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
def chunk_llm_docs(df: pd.DataFrame, base_model_name) -> pd.DataFrame:
    start_time = time.time()
    _, tokenizer = create_model_and_tokenizer(base_model_name)

    all_rows = []
    for _, row in df.iterrows():
        original_text = row["context"]
        # Overlap chunking: 512 max tokens, 256 stride
        chunked_texts = chunk_text_by_tokens(original_text, tokenizer,
                                             max_tokens=512, overlap=256)
        for idx, chunk in enumerate(chunked_texts):
            new_row = row.copy()
            new_row["context"] = chunk
            new_row["chunk_id"] = f"{row['context_id']}_{idx}"
            all_rows.append(new_row)

    elapsed = time.time() - start_time
    print(f'Chunking took {elapsed:.2f} sec, produced {len(all_rows)} rows.')

    wandb.init(project="MediMaven-LLM_finetuning",
               job_type="llm_finetune_pipeline",
               reinit=True)
    wandb.log({
        "num_original_docs": df.shape[0],
        "num_new_docs": len(all_rows),
        "chunking_time_sec": elapsed
    })
    wandb.finish()

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
    base_model_name: str,
    max_length: int
) -> Dataset:
    """
    Tokenizes the instruction text + answer for Causal LM.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
        
    _, tokenizer = create_model_and_tokenizer(base_model_name)
    
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
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# --------------------------------------------------------------
#   STEP 5: Train Model with Basic Param Tuning
# --------------------------------------------------------------
@step(enable_cache=False)
def train_model(
    base_model_name: str,
    split_ds: Dict[str, Any],
    max_len: int,
    output_dir: str = "./models/llama_finetuned",
    # We will do a mini param search:
    lr: float = 5e-5,
    lora_r: int= 32,
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
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_lora = get_peft_model(model, lora_config)

    # SFT Trainer config
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=max_len,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        save_strategy="epoch",
        logging_steps=1000,
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

    wandb.init(
        project="MediMaven-LLM-Finetuning",
        job_type="finetune",
        name=f"llama_finetune_{run_name}",
        reinit=True
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
@step
def merge_and_evaluate(
    best_model_path: str,
    base_model_name: str,
    heldout_ds
) -> float:
    """
    1) Load base model + LoRA adapter from best_model_path,
       merge them into a single set of weights.
    2) Evaluate perplexity on the held-out test set (never used in training or val).
    3) Optionally push to HF Hub or store locally for production usage.
    """
    import wandb
    from peft import PeftModel

    heldout_ds = heldout_ds["test"]
    # 1) Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True
    )
    # 2) Load LoRA and merge
    lora_model = PeftModel.from_pretrained(
        base_model,
        best_model_path,
        device_map="auto",
    )
    merged_model = lora_model.merge_and_unload()
    # Now merged_model has all weights, no need to load LoRA again.

    # Save or push to Hugging Face (commented out – example usage only):
    # merged_model.push_to_hub("YourUser/YourMergedModelName")

    # Evaluate perplexity on held-out set
    merged_model.eval()

    test_losses = []
    for example in heldout_ds:
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)
        input_ids = input_ids.to(merged_model.device)
        with torch.no_grad():
            outputs = merged_model(input_ids, labels=input_ids)
        test_losses.append(outputs.loss.item())

    avg_loss = sum(test_losses) / len(test_losses)
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    print(f"Held-Out Test Perplexity: {ppl:.3f}")

    # Optionally log to W&B
    wandb.init(project="MediMaven-LLM-Finetuning",
               job_type="merge_and_eval",
               name="final_heldout_eval",
               reinit=True)
    wandb.log({"heldout_ppl": ppl})
    wandb.finish()

    return ppl

# --------------------------------------------------------------
#   PIPELINE DEFINITION
# --------------------------------------------------------------
@pipeline  # Disable caching temporarily for debugging
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
    best_model_dir = train_model(
        base_model_name=MODEL_NAME,
        split_ds=split_ds,
        max_len=max_len,
        output_dir=output_dir,
        lr=5e-5,
        lora_r=32,
        epochs=2
    )
    
    # 7) Merge & Evaluate
    merge_and_evaluate(
        best_model_path=best_model_dir,
        base_model_name=MODEL_NAME,
        heldout_ds=split_ds
    )
    
    print("\n=== Pipeline Completed ===")