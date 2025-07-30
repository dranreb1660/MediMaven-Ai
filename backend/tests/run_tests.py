#!/usr/bin/env python
"""Universal test runner that works in all environments"""
import subprocess
import sys
import os

def run_tests():
    # Detect environment
    is_colab = os.path.exists("/content/drive")
    is_ci = os.environ.get("CI") == "true"
    
    print(f"Environment: {'Colab' if is_colab else 'CI' if is_ci else 'Local'}")
    
    # Set appropriate test command
    if is_colab:
        # Run Colab-specific tests
        cmd = ["python", "scripts/quick_test_colab.py"]
    else:
        # Run standard pytest
        test_files = [
            "tests/test_minimal.py",
            "tests/test_api_universal.py::TestSchemas",
            "tests/test_services_universal.py::TestBasicUnits",
        ]
        cmd = ["pytest"] + test_files + ["-v"]
    
    # Run tests
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())