#!/bin/bash

# 🗃️ Base directory for all models (shared between rag-api and tgi)
MODEL_DIR="./models"
GPTQ_DIR="$MODEL_DIR/llama_finetuned/gptqmodel_4bit"

# 🔗 Base URL for model files
BASE_URL="https://huggingface.co/dranreb1660/medimaven-models/resolve/main"

# 📝 Files needed for GPTQ Model (TGI)
GPTQ_FILES=(
  "config.json"
  "generation_config.json"
  "model-00001-of-00002.safetensors"
  "model-00002-of-00002.safetensors"
  "model.safetensors.index.json"
  "quant_log.csv"
  "quantize_config.json"
  "special_tokens_map.json"
  "tokenizer_config.json"
  "tokenizer.json"
)

# 📝 Extra files required by rag-api (faiss indexes, ranking model)
EXTRA_FILES=(
  "faiss_index.bin"
  "faiss_index_data.json"
  "faiss_index_mapping.json"
  "ltr_best_model.json"
)

# ✅ Ensure base directories exist
mkdir -p "$GPTQ_DIR"
chmod -R 777 "$MODEL_DIR"

# 📥 Download GPTQ files
echo "⬇️ Downloading GPTQ model files..."
for FILE in "${GPTQ_FILES[@]}"; do
  DEST="$GPTQ_DIR/$FILE"
  if [ ! -f "$DEST" ]; then
    echo "   📦 Fetching $FILE..."
    wget -q --show-progress "$BASE_URL/llama_finetuned/gptqmodel_4bit/$FILE" -O "$DEST"
  else
    echo "   ✅ $FILE already exists, skipping..."
  fi
done

# 📥 Download extra files
echo "⬇️ Downloading extra model files..."
for FILE in "${EXTRA_FILES[@]}"; do
  DEST="$MODEL_DIR/$FILE"
  if [ ! -f "$DEST" ]; then
    echo "   📦 Fetching $FILE..."
    wget -q --show-progress "$BASE_URL/$FILE" -O "$DEST"
  else
    echo "   ✅ $FILE already exists, skipping..."
  fi
done

echo "🎉 All models downloaded successfully!"
