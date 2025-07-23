#!/usr/bin/env python
"""Quick test runner to verify backend tests are working"""

import subprocess
import sys
import os

def run_backend_tests():
    """Run backend tests with proper environment setup"""
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['DATABASE_URL'] = 'sqlite:///test.db'
    env['ENABLE_MONITORING'] = 'false'
    env['JWT_SECRET'] = 'test-secret-key'
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    os.chdir(backend_dir)
    
    print("Running backend tests...")
    print(f"Working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    
    # Run pytest with minimal output
    cmd = [sys.executable, '-m', 'pytest', '-v', '--tb=short', '-x']
    
    result = subprocess.run(cmd, env=env)
    
    return result.returncode

if __name__ == '__main__':
    exit_code = run_backend_tests()
    sys.exit(exit_code)
