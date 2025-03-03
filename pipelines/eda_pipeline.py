from zenml.pipelines import pipeline
from zenml.steps import step
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, UTC



import os
import subprocess
from pipelines.eda import print_overview, plot_missing_values, plot_dataset_distribution, plot_speciality_distribution, plot_length_distributions, drop_invalid_entries, frequency_stats, plot_wordcloud, plot_correlation

# Define file paths for processed data and visualizations
viz_path = './data/logs/'
data_path_processed = "./data/processed/"

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["medimaven_db"]
raw_qa_collection = db["qa_master_raw"]
processed_qa_collection = db['qa_master_processed']

def ensure_mongodb_running():
    """Checks if MongoDB is running, and starts it if not."""
    try:
        # Try connecting to MongoDB
        subprocess.run(["mongosh", "--eval", "db.runCommand({ ping: 1 })"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ MongoDB is already running.")
    except subprocess.CalledProcessError:
        print("⚠️ MongoDB is NOT running. Attempting to start it...")
        os.system("brew services start mongodb-community")
        print("✅ MongoDB is running now!!!")


@step
def fetch_data():
    """Load data from MongoDB into Pandas DataFrame."""
    cursor = processed_qa_collection.find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    return df

@step
def perform_eda(df: pd.DataFrame):
    """
    Main function to perform the full EDA on the qa_master dataset.
    """
    # # Load the processed dataset
    # df = fetch_data()
    
    # Print basic overview of the dataset
    print_overview(df)
    
    # Plot missing values summary
    plot_missing_values(df, viz_path)
    
    # Plot the distribution of datasets (bar and pie charts)
    plot_dataset_distribution(df, viz_path)
    
    # Plot the distribution of specialities
    plot_speciality_distribution(df, viz_path)
    
    # Plot histograms and box plots for text length distributions
    plot_length_distributions(df, viz_path)
    
    # Drop invalid entries based on word count for 'answer' and 'context'
    df_cleaned = drop_invalid_entries(df)
    
    # Print frequency counts for 'focus' and 'qtype'
    frequency_stats(df_cleaned)
    
    # Generate and plot a word cloud of the most common words in questions
    plot_wordcloud(df_cleaned, viz_path)
    
    # Plot a heatmap showing correlation between question and answer lengths
    plot_correlation(df_cleaned, viz_path)


@step
def load_data(combined_df: pd.DataFrame):
    """Store cleaned data in MongoDB."""
    # Convert to list of dictionaries for MongoDB insertion
    records = combined_df.to_dict(orient="records")

    for record in records:
      record["tags"] = []  # Optional: You can generate tags if needed
      record["created_at"] = datetime.now(UTC)
      record["updated_at"] = datetime.now(UTC)      
            

    if records:
      processed_qa_collection.insert_many(records)
      print(f"Inserted {len(records)} processed medical Q&A records into MongoDB.")


@pipeline
def eda_pipeline():
    ensure_mongodb_running()
    df = fetch_data()
    perform_eda(df)

