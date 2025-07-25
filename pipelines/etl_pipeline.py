import pandas as pd
import re
from datetime import datetime, UTC
from zenml.pipelines import pipeline
from zenml.steps import step
from typing import Tuple
from src.backend.utils import ensure_mongodb_running, get_mongo_connection


# MongoDB Connection
db = get_mongo_connection()
medical_qa_collection = db["qa_master_raw"]


import os
import subprocess

def ensure_mongodb_running():
    """Checks if MongoDB is running, and starts it if not."""
    try:
        # Try connecting to MongoDB
        subprocess.run(["mongosh", "--eval", "db.runCommand({ ping: 1 })"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ MongoDB is already running.")
    except subprocess.CalledProcessError:
        print("⚠️ MongoDB is NOT running. Attempting to start it...")
        os.system("brew services start mongodb-community")
        print("✅ MongoDB is running")


def clean_text(text):
    """Remove HTML tags and clean up text formatting"""
    # Strip HTML tags and normalize whitespace
    text = re.sub(r"<.*?>", "", str(text))
    text = re.sub(r"[^a-zA-Z0-9.,!?\s]", " ", text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()



@step
def extract_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV data."""
    df_medquad = pd.read_csv("data/processed/medquad.csv")
    df_icliniq = pd.read_csv("data/processed/icliniq.csv")
    print('created df')
    return df_medquad, df_icliniq

@step
def transform_data(df_medquad: pd.DataFrame, df_icliniq: pd.DataFrame):
    """Clean and merge MedQuad and iCliniQ datasets"""

    # Clean MedQuad data
    print('MedQuad Data info:\n')
    print(df_medquad.isna().sum().reset_index())
    df_medquad.dropna(subset=['answer'], inplace=True)
    df_medquad['context'] = (
        df_medquad['synonyms'].fillna('') + ' ' +
        df_medquad['focus'].fillna('') + ' ' +
        df_medquad['question'].fillna('')
    )
    df_medquad['Dataset'] = 'MedQuad'
    df_medquad = df_medquad[['Dataset', 'focus', 'synonyms', 'qtype', 'question', 'context', 'answer']]

    # Clean iCliniQ data
    print('iClinique Data info:\n')
    print(df_icliniq.isna().sum().reset_index())
    df_icliniq['context'] = df_icliniq['Abstract'].fillna('') + ' ' + df_icliniq['Question'].fillna('')
    df_icliniq = df_icliniq[['Speciality', 'Title', 'context', 'Answer']]
    df_icliniq.rename(columns={
        "Speciality": "speciality",
        "Title": "question",
        "Answer": "answer"
    }, inplace=True)
    df_icliniq['Dataset'] = 'iCliniQ'

    # Merge datasets and clean text
    df_combined = pd.concat([df_medquad, df_icliniq], ignore_index=True)
    df_combined = df_combined.fillna('')
    
    df_combined["question"] = df_combined["question"].apply(clean_text)
    df_combined["answer"]   = df_combined["answer"].apply(clean_text)
    df_combined["context"]  = df_combined["context"].apply(clean_text)

    # Remove duplicates
    df_combined.drop_duplicates(subset=["question", "answer"], inplace=True)

    return df_combined

@step
def load_data(combined_df: pd.DataFrame):
    """Store cleaned data in MongoDB."""
    records = combined_df.to_dict(orient="records")

    for record in records:
      record["tags"] = []
      record["created_at"] = datetime.now(UTC)
      record["updated_at"] = datetime.now(UTC)
            

    if records:
      medical_qa_collection.insert_many(records)
      print(f"Inserted {len(records)} medical Q&A records into MongoDB.")


@pipeline
def etl_pipeline():
    ensure_mongodb_running()
    raw_medquad, raw_icliniq = extract_data()
    cleaned_data = transform_data(raw_medquad, raw_icliniq)
    load_data(cleaned_data)




# if __name__ == '__main__':
#     # Call this at the beginning of the script
#   ensure_mongodb_running()


