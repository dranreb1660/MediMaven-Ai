"""
One-command test runner for Google Colab
Just run: !python quick_test.py
"""
import subprocess
import sys
import os
import warnings

# Suppress AutoAWQ deprecation warning
warnings.filterwarnings('ignore', category=DeprecationWarning, module='awq')

# Quick setup and run
print("🚀 MediMaven Quick Test Runner\n")

# Install dependencies
print("📦 Installing test dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "aiosqlite", "pytest", "pytest-asyncio", "httpx"])

# Set environment
os.environ.update({
    "DATABASE_URL": "sqlite+aiosqlite:///test.db",
    "ENABLE_MONITORING": "false",
    "JWT_SECRET": "test-secret",
    "TESTING": "true",
    "PYTHONPATH": "/content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven"
})

# Change to backend directory
os.chdir("/content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven/backend")

print("\n✅ Running tests that work in Colab...\n")

# Run only tests that should pass
test_commands = [
    "pytest tests/test_minimal.py -v -q",
    "pytest tests/test_api.py::TestSchemas -v -q",
    "pytest tests/test_services.py::TestBasicUnits -v -q",
]

for cmd in test_commands:
    print(f"\n{'='*50}")
    print(f"Running: {cmd}")
    print('='*50)
    subprocess.run(cmd.split())

print("\n✅ Quick tests complete!")
print("\nFor full test details, run:")
print("  !cd backend && python tests/run_colab_tests.py")
