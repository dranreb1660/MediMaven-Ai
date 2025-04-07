#!/bin/bash

set -e

echo "📦 Installing rclone..."
apt update && apt install -y rclone fuse

echo "🗂️ Setting up config..."
mkdir -p ~/.config/rclone

cat <<EOF > ~/.config/rclone/rclone.conf
[gdrive]
type = drive
client_id = 594053889523-ka4uh3apmigkj869pn1uqnabsuitgv97.apps.googleusercontent.com
client_secret = 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
scope = drive
token = {"access_token":"ya29.a0AZYkNZiURkBh9oF3x_5jEheBRJ-sijANLG2IvQwiYTl_S1v7CAeUiyYslz3JzDcInxuX8BkFqmbuQc0fZOPUAnZ4Z7IELKXF_nhV9lxle0THXJFNtt9gPAHK2ujfa5yvE8HciP-OaHolklUVFri4x3fHHVIC_yhqmJNn6SU2aCgYKAR8SARASFQHGX2MiU_ThR72UB596nQYZUFGa3A0175","token_type":"Bearer","refresh_token":"1//01HJEIfUYxo9ECgYIARAAGAESNwF-L9Ir5IsGxDGxCNhiwd3zm4jeuMHKXXJr_XP4qBEL5FsDU5cXuGcZNbD1ipq6CR30XK-ktDA","expiry":"2025-04-04T17:56:42.789014-04:00"}
team_drive = 

EOF

echo "📂 Mounting Google Drive to ~/drive..."
mkdir -p ~/drive
chmod 775 ~/drive
rclone mount gdrive: ~/drive 
# rclone mount gdrive: ~/drive --vfs-cache-mode writes &


echo "🛠 Updating packages..."
apt-get install sudo
sudo apt update && sudo apt upgrade -y

echo "🐳 Installing Docker..."
sudo apt install -y docker.io

echo "🔌 Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

echo "🔁 Adding user to docker group..."
sudo usermod -aG docker $USER

echo "✅ All set! Log out and back in for docker group to apply."
echo "🔄 Rebooting system.. if you want"
n
