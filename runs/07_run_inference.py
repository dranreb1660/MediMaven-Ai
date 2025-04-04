from pipelines.rag_inference_pipeline import rag_inference_pipeline

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run inference on a question.")
    parser.add_argument('-q', '--question', type=str, required=True, help='Question to ask the model')
    
    args = parser.parse_args()
    response = rag_inference_pipeline(args.question)
    print(response)

if __name__ == "__main__":
    main()

    