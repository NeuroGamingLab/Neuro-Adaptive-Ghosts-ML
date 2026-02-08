#!/bin/bash

# LocalStack configuration
LOCALSTACK_ENDPOINT="http://localhost:4566"
BUCKET_NAME="pacman-game"
REGION="us-east-1"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Deploying Pacman Game to LocalStack...${NC}"

# Check if LocalStack is running
echo -e "${YELLOW}Checking if LocalStack is running...${NC}"
if ! curl -s "${LOCALSTACK_ENDPOINT}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: LocalStack is not running on ${LOCALSTACK_ENDPOINT}${NC}"
    echo -e "${YELLOW}Please start LocalStack first:${NC}"
    echo "  docker run -d -p 4566:4566 localstack/localstack"
    exit 1
fi

echo -e "${GREEN}LocalStack is running!${NC}"

# Configure AWS CLI to use LocalStack
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=${REGION}

# Create S3 bucket if it doesn't exist
echo -e "${YELLOW}Creating S3 bucket...${NC}"
aws --endpoint-url=${LOCALSTACK_ENDPOINT} s3 mb s3://${BUCKET_NAME} 2>/dev/null || echo "Bucket may already exist"

# Enable static website hosting
echo -e "${YELLOW}Configuring static website hosting...${NC}"
aws --endpoint-url=${LOCALSTACK_ENDPOINT} s3 website s3://${BUCKET_NAME} \
    --index-document index.html \
    --error-document index.html

# Upload files to S3
echo -e "${YELLOW}Uploading game files...${NC}"
aws --endpoint-url=${LOCALSTACK_ENDPOINT} s3 cp index.html s3://${BUCKET_NAME}/index.html --content-type "text/html"
aws --endpoint-url=${LOCALSTACK_ENDPOINT} s3 cp game.js s3://${BUCKET_NAME}/game.js --content-type "application/javascript"
aws --endpoint-url=${LOCALSTACK_ENDPOINT} s3 cp style.css s3://${BUCKET_NAME}/style.css --content-type "text/css"

# Set public read permissions (for LocalStack)
echo -e "${YELLOW}Setting bucket policy...${NC}"
cat > /tmp/bucket-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::${BUCKET_NAME}/*"
    }
  ]
}
EOF

aws --endpoint-url=${LOCALSTACK_ENDPOINT} s3api put-bucket-policy \
    --bucket ${BUCKET_NAME} \
    --policy file:///tmp/bucket-policy.json

# Get the website endpoint
WEBSITE_ENDPOINT="${LOCALSTACK_ENDPOINT}/${BUCKET_NAME}/index.html"

echo -e "${GREEN}✓ Deployment complete!${NC}"
echo ""
echo -e "${GREEN}Game URL:${NC} ${WEBSITE_ENDPOINT}"
echo ""
echo -e "${YELLOW}Note:${NC} LocalStack S3 doesn't support website hosting endpoints by default."
echo -e "You can access the game directly at: ${WEBSITE_ENDPOINT}"
echo ""
echo -e "${YELLOW}Alternative:${NC} Use a simple HTTP server to serve the files:"
echo "  python3 -m http.server 8000"
echo "  Then open: http://localhost:8000"
