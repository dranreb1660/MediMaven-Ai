import pandas as pd
from pathlib import Path
import re

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



def merge_clean_datasets():
    """
    Read, clean, merge, and save the MedQuad and iCliniQ datasets.

    This function performs the following steps:
      - Reads processed CSV files for MedQuad and iCliniQ.
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
      - Saves the final merged DataFrame to a CSV file.
    """

    # Read processed CSV files for both datasets
    df_medquad = pd.read_csv("data/processed/medquad.csv")
    df_icliniq = pd.read_csv("data/processed/icliniq.csv")

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
    print(df_medquad.isna().sum().reset_index())
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

    # Save the final merged DataFrame to a CSV file
    csv_output_path = "data/processed/qa_master.csv"
    df_combined.to_csv(csv_output_path, index=False)
    print(f'Csv saved to {csv_output_path}')

if __name__ == '__main__':
    merge_clean_datasets()

