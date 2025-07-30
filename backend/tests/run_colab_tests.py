#!/usr/bin/env python
"""
Quick test runner for Google Colab
Runs only tests that work without full environment setup
"""
import subprocess
import os
import sys

def run_colab_tests():
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Set up environment
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"
    os.environ["ENABLE_MONITORING"] = "false"
    os.environ["JWT_SECRET"] = "test-secret-key"
    os.environ["TESTING"] = "true"
    
    # Ensure we're in the right directory
    backend_dir = "/content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven/backend"
    if os.path.exists(backend_dir):
        os.chdir(backend_dir)
    
    # Add to Python path
    sys.path.insert(0, os.path.dirname(backend_dir))
    
    print("MediMaven Test Runner - Colab Edition")
    print("=" * 50)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path includes: {sys.path[0]}")
    print()
    
    # Install aiosqlite if needed
    try:
        import aiosqlite
    except ImportError:
        print("Installing aiosqlite...")
        subprocess.run([sys.executable, "-m", "pip", "install", "aiosqlite", "-q"])
    
    # Define test groups that should work in Colab
    test_groups = [
        ("Basic Tests", "tests/test_minimal.py -v"),
        ("Schema Tests", "tests/test_api.py::TestSchemas -v"),
        ("Basic Unit Tests", "tests/test_services.py::TestBasicUnits -v"),
        ("Mock Integration", "tests/test_services.py::TestMockIntegration -v"),
        ("Cache Tests", "tests/test_services.py::TestCaching -v"),
    ]
    
    passed = 0
    failed = 0
    
    for group_name, test_cmd in test_groups:
        print(f"\n{'='*50}")
        print(f"Running: {group_name}")
        print(f"{'='*50}")
        
        cmd = [sys.executable, "-m", "pytest"] + test_cmd.split()
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {group_name} PASSED")
            passed += 1
        else:
            print(f"❌ {group_name} FAILED")
            failed += 1
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    
    return failed == 0

if __name__ == "__main__":
    success = run_colab_tests()
    sys.exit(0 if success else 1)
