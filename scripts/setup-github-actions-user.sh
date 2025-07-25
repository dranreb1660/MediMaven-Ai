#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 Setting up GitHub Actions IAM user for MediMaven deployment${NC}"

# Variables
USER_NAME="medimaven-github-actions"
POLICY_NAME="MediMavenDeploymentPolicy"
POLICY_FILE="docs/aws-iam-policy.json"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not configured. Run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Current AWS identity:${NC}"
aws sts get-caller-identity

# Create IAM user
echo -e "\n${GREEN}👤 Creating IAM user: $USER_NAME${NC}"
if aws iam get-user --user-name $USER_NAME &> /dev/null; then
    echo -e "${YELLOW}⚠️  User $USER_NAME already exists${NC}"
else
    aws iam create-user --user-name $USER_NAME
    echo -e "${GREEN}✅ User created successfully${NC}"
fi

# Create and attach policy
echo -e "\n${GREEN}📜 Creating IAM policy: $POLICY_NAME${NC}"
if aws iam get-policy --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$POLICY_NAME" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Policy $POLICY_NAME already exists${NC}"
else
    POLICY_ARN=$(aws iam create-policy \
        --policy-name $POLICY_NAME \
        --policy-document file://$POLICY_FILE \
        --query 'Policy.Arn' \
        --output text)
    echo -e "${GREEN}✅ Policy created: $POLICY_ARN${NC}"
fi

# Get policy ARN (in case it already existed)
POLICY_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$POLICY_NAME"

# Attach policy to user
echo -e "\n${GREEN}🔗 Attaching policy to user${NC}"
aws iam attach-user-policy --user-name $USER_NAME --policy-arn $POLICY_ARN
echo -e "${GREEN}✅ Policy attached successfully${NC}"

# Create access keys
echo -e "\n${GREEN}🔑 Creating access keys${NC}"
KEY_OUTPUT=$(aws iam create-access-key --user-name $USER_NAME --output json)

ACCESS_KEY_ID=$(echo $KEY_OUTPUT | jq -r '.AccessKey.AccessKeyId')
SECRET_ACCESS_KEY=$(echo $KEY_OUTPUT | jq -r '.AccessKey.SecretAccessKey')

echo -e "\n${GREEN}🎉 Setup complete!${NC}"
echo -e "\n${YELLOW}📋 Add these secrets to your GitHub repository:${NC}"
echo -e "${GREEN}AWS_ACCESS_KEY_ID:${NC}     $ACCESS_KEY_ID"
echo -e "${GREEN}AWS_SECRET_ACCESS_KEY:${NC} $SECRET_ACCESS_KEY"

echo -e "\n${YELLOW}🔒 IMPORTANT SECURITY NOTES:${NC}"
echo -e "• Store these credentials securely - they won't be shown again"
echo -e "• Add them to GitHub Secrets (Settings → Secrets and variables → Actions)"
echo -e "• Never commit these credentials to your repository"
echo -e "• Consider enabling MFA for additional security"

echo -e "\n${GREEN}📖 Next steps:${NC}"
echo -e "1. Copy the credentials above"
echo -e "2. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions"
echo -e "3. Add both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY secrets"
echo -e "4. Test your deployment workflow"
