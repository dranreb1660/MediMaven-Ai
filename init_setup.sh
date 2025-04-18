#!/bin/bash

set -e

echo "📦 Installing rclone..."
apt update && apt install -y rclone fuse

echo "🗂️ Setting up config..."
mkdir -p ~/.config/rclone

cat <<EOF > ~/.config/rclone/rclone.conf
[[gdrive]
type = drive
scope = drive
token = {"access_token":"ya29.a0AZYkNZg0Ays_kexXD92F4EG5ja260X_aCI43-Eq--543xK9DcV8jwRTjRMgsypVAdrj6EtPRzqggVxfa6ALiYj6mD7ZZWJcY0uPyHxhpqJ7SITkWaV4clKG7NzyDP_bO3x8NVbXeky1qV4bYURhnFz45LR-IRQeClciAYj-2aCgYKAeISARASFQHGX2Miah3Y91nszUvTv2F3DP1_uw0175","token_type":"Bearer","refresh_token":"1//01HeSF5pu9kGiCgYIARAAGAESNwF-L9Ir5n2r9vEMp2J192bqg7DuYzbFgmamd2yFGVNsVcj2QWtIfDJrNfEuZsPdJ1udr-BaTW4","expiry":"2025-04-04T15:45:30.546069-04:00"}
team_drive = 
