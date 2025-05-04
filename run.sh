echo "🔐 Please enter your Weights & Biases API key:"
read -s WANDB_API_KEY

echo "🔐 Please enter your Hugging Face token (optional, hit enter to skip):"
read -s HF_TOKEN

# Write to a temporary .env file (or persist it if you prefer)
cat <<EOF > .env.temp
WANDB_API_KEY=${WANDB_API_KEY}
HF_TOKEN=${HF_TOKEN}
EOF
cat .env.temp
export WANDB_API_KEY
export HF_TOKEN
# Use it for the run
sudo docker compose -f docker-compose.prod.demo.yml up
