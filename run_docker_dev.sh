#!/bin/bash
# =============================================================================
# run_docker_dev.sh - Run FacadeAI in Docker development environment
# =============================================================================
# This script builds and runs a Docker container for FacadeAI development
# =============================================================================
set -e

# Build and start the Docker container
echo "Building and starting FacadeAI development Docker container..."
docker-compose -f docker-compose.dev.yml up -d

# Display information
echo ""
echo "FacadeAI development environment is running."
echo ""
echo "To access the container:"
echo "  docker exec -it testbuild_facadeai-dev_1 bash"
echo ""
echo "Once inside the container, you can:"
echo "  - Run the verification script: ./verify_env.py"
echo "  - Run the inference script: ./dev_runlocal.sh"
echo ""
echo "To stop the container:"
echo "  docker-compose -f docker-compose.dev.yml down"
echo ""