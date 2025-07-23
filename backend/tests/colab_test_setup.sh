#!/bin/bash
# Quick setup script for running tests in Google Colab

echo "Setting up test environment for Colab..."

# Set Python path
export PYTHONPATH="/content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven:$PYTHONPATH"

# Create test database
export DATABASE_URL="sqlite:///test.db"
export ENABLE_MONITORING="false"
export JWT_SECRET="test-secret-key"
export OPENAI_API_KEY="test-key"

# Change to backend directory
cd /content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven/backend

# Run only the minimal tests first
echo "Running minimal tests..."
python -m pytest tests/test_minimal.py -v

# If minimal tests pass, try basic tests
if [ $? -eq 0 ]; then
    echo "Running basic tests..."
    python -m pytest tests/test_basic.py -v
fi

# Show summary
echo "Test setup complete. To run all tests:"
echo "  pytest -v"
echo "To run specific test file:"
echo "  pytest tests/test_api.py -v"
