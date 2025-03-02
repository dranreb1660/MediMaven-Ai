#!/usr/bin/env python
"""
run_dag.py - A utility script to run Airflow DAG files while suppressing specific deprecation warnings.

Usage:
    python run_dag.py [options] <path_to_dag_file>

Options:
    --use-subprocess  Run the DAG file in a subprocess to ensure environment variables are properly set

Example:
    python run_dag.py airflow_etl/dags/medical_qa_etl_dag.py
    python run_dag.py --use-subprocess airflow_etl/dags/medical_qa_etl_dag.py
"""

import warnings
import sys
import os
import importlib.util
import runpy
import traceback
import subprocess
import argparse


def set_warning_environment():
    """Set environment variables to suppress specific deprecation warnings."""
    # Set PYTHONWARNINGS to ignore the utcnow() deprecation warnings
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:airflow.utils.log.file_processor_handler:,ignore::DeprecationWarning:airflow.utils.timezone:"
    
    # Also set up the Python warnings filter as a fallback
    warnings.filterwarnings(
        "ignore",
        message="datetime\\.datetime\\.utcnow\\(\\) is deprecated",
        category=DeprecationWarning,
    )
    
    # You can add more specific warning filters here if needed


def run_dag_file(dag_file_path, use_subprocess=False):
    """Execute the specified DAG file.
    
    Args:
        dag_file_path: Path to the DAG file to run
        use_subprocess: If True, run the DAG file in a subprocess to ensure
                        environment variables are properly set
    """
    if not os.path.exists(dag_file_path):
        print(f"Error: DAG file not found: {dag_file_path}")
        return False
    
    if use_subprocess:
        # Run in subprocess to ensure environment variables are set correctly
        try:
            # Create a new environment with our warning settings
            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "ignore::DeprecationWarning:airflow.utils.log.file_processor_handler:,ignore::DeprecationWarning:airflow.utils.timezone:"
            
            # Run the DAG file in a subprocess
            result = subprocess.run(
                [sys.executable, dag_file_path],
                env=env,
                check=False,
                text=True,
                capture_output=True
            )
            
            # Output results
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error running DAG file in subprocess: {e}")
            print("\nTraceback:")
            traceback.print_exc()
            return False
    else:
        # Run in the current process
        try:
            # Method 1: Using runpy (simulates running a script like python file.py)
            runpy.run_path(dag_file_path, run_name="__main__")
            return True
        except Exception as e:
            print(f"Error running DAG file: {e}")
            print("\nTraceback:")
            traceback.print_exc()
            return False


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Airflow DAG files while suppressing specific deprecation warnings."
    )
    parser.add_argument(
        "dag_file_path",
        help="Path to the DAG file to run"
    )
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        help="Run the DAG file in a subprocess to ensure environment variables are properly set"
    )
    
    # Parse arguments and handle help/usage display
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
        
    args = parser.parse_args()
    
    # Set up warning environment (for in-process execution)
    set_warning_environment()
    
    print(f"Running DAG file: {args.dag_file_path} (with datetime.utcnow() warnings suppressed)")
    print(f"Mode: {'subprocess' if args.use_subprocess else 'in-process'}")
    
    # Run the DAG file
    success = run_dag_file(args.dag_file_path, args.use_subprocess)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

