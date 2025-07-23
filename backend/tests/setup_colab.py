"""Test setup script for Google Colab"""
import subprocess
import sys
import os

def setup_colab_tests():
    """Setup test environment for Google Colab"""
    
    print("Setting up MediMaven tests for Google Colab...")
    
    # Install async SQLite driver
    print("Installing aiosqlite...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiosqlite"])
    
    # Set environment variables
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"
    os.environ["ENABLE_MONITORING"] = "false"
    os.environ["JWT_SECRET"] = "test-secret-key"
    os.environ["TESTING"] = "true"
    os.environ["PYTHONPATH"] = "/content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven"
    
    print("\nEnvironment configured:")
    print(f"  DATABASE_URL: {os.environ['DATABASE_URL']}")
    print(f"  PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    # Change to backend directory
    backend_dir = "/content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven/backend"
    os.chdir(backend_dir)
    print(f"\nWorking directory: {os.getcwd()}")
    
    print("\nSetup complete! You can now run:")
    print("  pytest tests/test_minimal.py -v    # For basic tests")
    print("  pytest tests/test_services.py -v   # For service tests")
    print("  pytest -v                          # For all tests")

if __name__ == "__main__":
    setup_colab_tests()
