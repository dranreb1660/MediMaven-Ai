import pandas as pd
import re
from datetime import datetime, UTC
from zenml.pipelines import pipeline
from zenml.steps import step
from typing import Tuple
from src.utils import ensure_mongodb_running, get_mongo_connection


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
    """
    Clean the input text by removing HTML tags and unwanted characters.
    This function converts the input to a string, removes any HTML tags,
    filters out characters that are not alphanumeric, punctuation (.,!?), 
    or whitespace, and trims leading and trailing whitespace.

    Args:
        text (str): The text to be cleaned.
    Returns:
        str: The cleaned text.
    """
    # Remove HTML tags from the text
    text = re.sub(r"<.*?>", "", str(text))
    # Remove characters that are not letters, numbers, punctuation, or whitespace
    text = re.sub(r"[^a-zA-Z0-9.,!?\s]", " ", text)
    #colapse 2 white spaces to one
    text = re.sub(r'\s{2,}', ' ', text)
    # Return the cleaned text with whitespace trimmed
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
    """
    clean, merge, and save the MedQuad and iCliniQ datasets.

    This function performs the following steps:
      - Cleans the MedQuad data by dropping rows with missing answers,
        creating a 'context' column from 'synonyms', 'focus', and 'question',
        and selecting the relevant columns.
      - Cleans the iCliniQ data by creating a 'context' column from 'Abstract'
        and 'Question', renaming columns to match MedQuad format, and selecting
        the relevant columns.
      - Merges both datasets into one DataFrame.
      - Applies text cleaning (using the `clean_text` function) to 'question',
        'answer', and 'context' columns.
      - Removes duplicate rows based on 'question' and 'answer'.
    """

    # ----- MedQuad Data Cleaning -----
    print('MedQuad Data info:\n')
    print(df_medquad.isna().sum().reset_index())
    # Drop rows where 'answer' is null
    df_medquad.dropna(subset=['answer'], inplace=True)
    # Create 'context' by concatenating 'synonyms', 'focus', and 'question'
    df_medquad['context'] = (
        df_medquad['synonyms'].fillna('') + ' ' +
        df_medquad['focus'].fillna('') + ' ' +
        df_medquad['question'].fillna('')
    )
    # Add a column indicating the dataset source
    df_medquad['Dataset'] = 'MedQuad'
    # Select the desired columns
    df_medquad = df_medquad[['Dataset', 'focus', 'synonyms', 'qtype', 'question', 'context', 'answer']]

    # ----- iCliniQ Data Cleaning -----
    print('iClinique Data info:\n')
    # Note: The following print currently prints MedQuad NA info again; adjust if needed.
    print(df_icliniq.isna().sum().reset_index())
    # Create 'context' by concatenating 'Abstract' and 'Question'
    df_icliniq['context'] = df_icliniq['Abstract'].fillna('') + ' ' + df_icliniq['Question'].fillna('')
    # Select the relevant columns from iCliniQ dataset
    df_icliniq = df_icliniq[['Speciality', 'Title', 'context', 'Answer']]
    # Rename columns to align with MedQuad dataset
    df_icliniq.rename(columns={
        "Speciality": "speciality",
        "Title": "question",
        "Answer": "answer"
    }, inplace=True)
    # Add a column indicating the dataset source
    df_icliniq['Dataset'] = 'iCliniQ'

    # ----- Merge Datasets -----
    # Concatenate the two DataFrames
    df_combined = pd.concat([df_medquad, df_icliniq], ignore_index=True)
    df_combined = df_combined.fillna('')
    # ----- Clean Text Columns -----
    # Apply the clean_text function to remove unwanted characters from text columns
    df_combined["question"] = df_combined["question"].apply(clean_text)
    df_combined["answer"]   = df_combined["answer"].apply(clean_text)
    df_combined["context"]  = df_combined["context"].apply(clean_text)


    # ----- Remove Duplicates -----
    # Drop duplicate rows based on the combination of 'question' and 'answer'
    df_combined.drop_duplicates(subset=["question", "answer"], inplace=True)

    return df_combined

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


