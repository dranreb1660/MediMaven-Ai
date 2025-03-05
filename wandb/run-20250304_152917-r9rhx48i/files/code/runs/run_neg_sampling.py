
from pipelines.neg_sampling_pipeline import (
    neg_sampling_pipeline,
    fetch_data_from_mongo,
    create_ltr_and_llm_datasets,
    store_and_log_results,
)

if __name__ == "__main__":
    # Configure your step params here:
    neg_sampling_pipeline()