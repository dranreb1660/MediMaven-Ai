#!/usr/bin/env python3
"""
MediMaven ML Pipeline Runner

Simple script to run the ML pipeline with different configurations.
This provides an easy entry point for training and inference.

My usage patterns:
- Full pipeline: python run_pipeline.py --stage full
- Individual stages: python run_pipeline.py --stage data_processing
- Custom config: python run_pipeline.py --config_dir custom_config/
"""

import sys
import pathlib
import argparse

# Add project root to path
project_root = pathlib.Path(__file__).parent
sys.path.append(str(project_root))

from pipelines.v1_1.main_pipeline import MediMavenPipeline


def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="MediMaven ML Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py --stage full
  
  # Run only data processing
  python run_pipeline.py --stage data_processing
  
  # Use custom config directory
  python run_pipeline.py --config_dir my_config/ --stage full
  
  # Run with debug logging
  python run_pipeline.py --stage full --log_level DEBUG
        """
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        choices=['data_processing', 'embedding_generation', 'negative_sampling', 'ltr_training', 'full'],
        default='full',
        help='Pipeline stage to run (default: full)'
    )
    
    parser.add_argument(
        '--config_dir',
        type=str,
        default='config',
        help='Directory containing configuration files (default: config)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    print("🚀 MediMaven ML Pipeline Runner")
    print(f"📁 Config directory: {args.config_dir}")
    print(f"🔧 Stage: {args.stage}")
    print(f"📊 Log level: {args.log_level}")
    print("-" * 50)
    
    try:
        # Initialize and run pipeline
        pipeline = MediMavenPipeline(config_dir=args.config_dir)
        
        if args.stage == 'full':
            results = pipeline.run_full_pipeline()
            
            print("\n✅ Pipeline completed successfully!")
            print("\n📈 Results Summary:")
            for stage, result in results.items():
                if stage != 'error':
                    print(f"  {stage}: ✓")
                    
                    # Show specific metrics for LTR training
                    if stage == 'ltr_training' and isinstance(result, dict):
                        training_info = result.get('training_info', {})
                        if 'best_score' in training_info:
                            print(f"    NDCG@10: {training_info['best_score']:.4f}")
                        if 'model_path' in result:
                            print(f"    Model: {result['model_path']}")
        else:
            pipeline.run_stage(args.stage)
            print(f"\n✅ Stage '{args.stage}' completed successfully!")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you have the required configuration files and data.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)
    
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
