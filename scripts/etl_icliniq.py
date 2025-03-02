import pandas as pd


def extract_icliniq():
    """
    Extract iCliniq data from a CSV file and select specific columns.

    This function reads the raw iCliniq CSV data from the specified path,
    selects a subset of columns, and returns the processed DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the selected columns ('Title', 'Title', 'Abstract', 'Question', 'Answer').
    """
    # Read the CSV file containing the raw iCliniq data
    df = pd.read_csv("./data/raw/icliniq/icliniq_data.csv")

    # Select the desired columns from the DataFrame
    df = df[['Speciality', 'Title', 'Abstract','Question', 'Answer' ]]

    return df

if __name__ == '__main__':
    csv_output = './data/processed/icliniq.csv'
    df_icliniq = extract_icliniq()
    df_icliniq.to_csv(csv_output, index=False)

    print(f'csv saved to {csv_output}')
