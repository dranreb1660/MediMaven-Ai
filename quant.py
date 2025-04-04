from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_path = "./models/llama_finetuned/merged_4bit"
tokenizer_path = "./models/llama_finetuned"
output_dir = "./models/llama_finetuned/autogptq"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

# Quantization configuration
quantization_config = BaseQuantizeConfig(
    bits=4,                # number of bits to quantize to
    group_size=128,        # group size
    desc_act=False,        # whether to use activation descriptors
    sym=False             # whether to enforce symmetric quantization
)

# Load and quantize the model
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config=quantization_config,
    trust_remote_code=True
)

# Save the quantized model
model.save_pretrained(
    output_dir,
    use_safetensors=True
)

# Save the tokenizer
tokenizer.save_pretrained(output_dir)