#!/bin/bash

# Exit on error
set -e

# Configuration
PROJECT_ID="rizeo-40249"      # Your GCP project ID
REGION="us-west2"             # Match the region where your VM is located
REPOSITORY="tarsier"          # Name of your Artifact Registry repository
IMAGE_NAME="tarsier-api"      # Name of your Docker image
TAG="latest"                  # Image tag

# Full image path
IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud is not installed. Please install the Google Cloud SDK first."
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed. Please install Docker first."
    exit 1
fi

# Ensure we're authenticated with GCP
echo "Ensuring GCP authentication..."
gcloud auth print-access-token &> /dev/null || {
    echo "Not authenticated with GCP. Running gcloud auth login..."
    gcloud auth login
}

# Create the Artifact Registry repository if it doesn't exist
echo "Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories create $REPOSITORY \
    --repository-format=docker \
    --location=$REGION \
    --description="Repository for Tarsier API" || {
    echo "Repository already exists or creation failed. Continuing..."
}

# Configure Docker to use Google Cloud as a credential helper
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_PATH . || {
    echo "Error: Docker build failed"
    exit 1
}

# Push the image to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push $IMAGE_PATH || {
    echo "Error: Failed to push image to Artifact Registry"
    exit 1
}

echo "Successfully built and pushed image: $IMAGE_PATH"
echo ""
echo "To pull and run this image on your VM:"
echo "1. Authenticate with GCP:"
echo "   gcloud auth login"
echo "   gcloud auth configure-docker ${REGION}-docker.pkg.dev"
echo ""
echo "2. Pull the image:"
echo "   docker pull $IMAGE_PATH"
echo ""
echo "3. Run the container:"
echo "   docker run --gpus all -p 8000:8000 --restart unless-stopped $IMAGE_PATH" 