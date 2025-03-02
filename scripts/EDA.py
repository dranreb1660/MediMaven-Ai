import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
import missingno as msno

# Define file paths for processed data and visualizations
viz_path = './data/logs/'
data_path_processed = "./data/processed/"

def load_data(file_path):
    """
    Load CSV data from the specified file path.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)

def print_overview(df):
    """
    Print basic overview of the DataFrame, including its shape and column names.
    
    Args:
        df (pd.DataFrame): DataFrame to overview.
    """
    print("DataFrame shape:", df.shape)
    print("\nColumns:\n", df.columns)

def plot_missing_values(df, save_path):
    """
    Plot a bar chart of missing values by column using missingno.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.
        save_path (str): Directory path where the plot image will be saved.
    """
    print("\nMissing Value Summary:\n", df.isnull().sum())
    plt.figure(figsize=(3, 1))
    msno.bar(df)
    plt.title("Missing Values by Column")
    plt.savefig(save_path + 'missing_values.png')
    plt.show()

def plot_dataset_distribution(df, save_path):
    """
    Plot the distribution of the 'Dataset' column using both bar and pie charts.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'Dataset' column.
        save_path (str): Directory path where the plot image will be saved.
    """
    dataset_counts = df['Dataset'].value_counts(dropna=False)
    print("\nDataset Distribution:\n", dataset_counts)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of dataset counts
    sns.countplot(data=df, x='Dataset', ax=axs[0])
    axs[0].set_title("Count of Rows by Dataset")
    
    # Pie chart of dataset distribution
    explodes = [0.1, 0]  # Explode the first slice for emphasis
    axs[1].pie(dataset_counts, labels={i: j for i, j in dataset_counts.items()},
               autopct='%1.1f%%', startangle=90, explode=explodes)
    axs[1].axis('equal')  # Ensure the pie is circular
    axs[1].set_title('Dataset Distribution: Source of Data')
    
    plt.tight_layout()
    plt.savefig(save_path + 'dataset_distribution.png')
    plt.show()

def plot_speciality_distribution(df, save_path):
    """
    Plot the distribution of the 'speciality' column as a horizontal bar chart.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'speciality' column.
        save_path (str): Directory path where the plot image will be saved.
    """
    speciality_counts = df['speciality'].value_counts(dropna=False)
    print("\nSpeciality Distribution:\n", speciality_counts.head(20))
    
    plt.figure(figsize=(14, 8))
    ax = speciality_counts.head(20).plot(kind='barh')
    plt.title("Top 20 Specialities")
    
    # Annotate each bar with its count
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 0),  # No offset
                    textcoords='data',
                    ha='center', va='bottom')
    
    plt.savefig(save_path + 'top_specialities.png')
    plt.show()

def add_stats(ax, data, color_mean='red', color_median='red'):
    """
    Add a vertical line for the median value and annotate it on a plot.
    
    Args:
        ax (matplotlib.axes.Axes): Axes object where the stats will be added.
        data (pd.Series): Data for which the median is computed.
        color_mean (str): Color for the mean line (not used in this function).
        color_median (str): Color for the median line.
    """
    mean_val = data.mean()
    median_val = data.median()
    # Draw a vertical line at the median value
    ax.axvline(median_val, color=color_median, linewidth=1)
    ylim = ax.get_ylim()[1]
    ax.text(median_val, ylim * 0.8, f'Median: {median_val:.1f}', color='b',
            va='top', ha='left')

def plot_length_distributions(df, save_path):
    """
    Plot histograms and box plots for text length metrics: question, answer, and context.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'question_length', 'answer_length', 
                           and 'context_length' columns.
        save_path (str): Directory path where the plot image will be saved.
    """
    
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    df = df.copy()
    df['question_length'] = df['question'].astype(str).apply(len)
    df['answer_length']   = df['answer'].astype(str).apply(len)
    df['context_length']   = df['context'].astype(str).apply(len)
    
    # Histogram plots (top row)
    sns.histplot(data=df, x='question_length', ax=axes[0, 0], bins=30, kde=False)
    axes[0, 0].set_title("Question Length Histogram")
    add_stats(axes[0, 0], df['question_length'])
    
    sns.histplot(data=df, x='answer_length', ax=axes[0, 1], bins=30, kde=False)
    axes[0, 1].set_title("Answer Length Histogram")
    add_stats(axes[0, 1], df['answer_length'])
    
    sns.histplot(data=df, x='context_length', ax=axes[0, 2], bins=30, kde=False)
    axes[0, 2].set_title("Context Length Histogram")
    add_stats(axes[0, 2], df['context_length'])
    
    # Box plots (bottom row) to highlight outliers
    sns.boxplot(x=df['question_length'], ax=axes[1, 0])
    axes[1, 0].set_title("Question Length Box Plot")
    add_stats(axes[1, 0], df['question_length'])
    
    sns.boxplot(x=df['answer_length'], ax=axes[1, 1])
    axes[1, 1].set_title("Answer Length Box Plot")
    add_stats(axes[1, 1], df['answer_length'])
    
    sns.boxplot(x=df['context_length'], ax=axes[1, 2])
    axes[1, 2].set_title("Context Length Box Plot")
    add_stats(axes[1, 2], df['context_length'])
    
    plt.tight_layout()
    plt.savefig(save_path + 'distribution_of_lengths.png')
    plt.show()

def drop_invalid_entries(df):
    """
    Drop rows from the DataFrame where text columns have an insufficient number of words.
    
    Specifically, drop rows where:
      - 'answer' has fewer than 6 words.
      - 'context' has fewer than 24 words.
    
    Also prints some debug information about entries that may be too short.
    
    Args:
        df (pd.DataFrame): DataFrame to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Print questions with less than 5 words (for inspection)
    print(df[df['question'].str.split().str.len() < 5].question)
    
    # Print answers with less than 5 words (for inspection)
    print(df[df['answer'].str.split().str.len() < 5].answer)
    
    # Drop rows where 'answer' has fewer than 6 words
    drop_indices = df[df['answer'].str.split().str.len() < 6].index
    df.drop(drop_indices, inplace=True)
    
    # Print remaining answers with less than 7 words (for inspection)
    print(df[df['answer'].str.split().str.len() < 7].answer)
    
    # Print context entries with less than 25 words (for inspection)
    print(df[df['context'].str.split().str.len() < 25].context)
    
    # Drop rows where 'context' has fewer than 24 words
    drop_indices = df[df['context'].str.split().str.len() < 24].index
    df.drop(drop_indices, inplace=True)
    
    # Update 'context_length' after cleaning
    df['context_length'] = df['context'].astype(str).apply(len)
    
    return df

def plot_wordcloud(df, save_path):
    """
    Generate and display a word cloud based on the text in the 'question' column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'question' column.
        save_path (str): Directory path where the word cloud image will be saved.
    """
    all_questions = " ".join(df['question'].dropna().astype(str).values)
    wordcloud = WordCloud(background_color="white", max_words=100).generate(all_questions)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Most Common Words in Questions")
    plt.savefig(save_path + 'key_words.png')
    plt.show()

def plot_correlation(df, save_path):
    """
    Plot a heatmap showing the correlation between text length metrics.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'question_length' and 'answer_length'.
        save_path (str): Directory path where the correlation heatmap image will be saved.
    """
    df = df.copy()
    df['question_length'] = df['question'].astype(str).apply(len)
    df['answer_length']   = df['answer'].astype(str).apply(len)
    df['context_length']   = df['context'].astype(str).apply(len)

    corr_matrix = df[['question_length', 'answer_length', 'context_length']].corr()
    print("\nCorrelation matrix (length columns):\n", corr_matrix)
    sns.heatmap(corr_matrix, annot=True, cmap="Blues")
    plt.title("Correlation of Question & Answer Length & Context length")
    plt.savefig(save_path + 'corelation.png')
    plt.show()

def frequency_stats(df):
    """
    Print frequency counts for the 'focus' and 'qtype' columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'focus' and 'qtype' columns.
    """
    focus_counts = df['focus'].value_counts(dropna=False)
    qtype_counts = df['qtype'].value_counts(dropna=False)
    print("\nFocus Frequency:\n", focus_counts.head())
    print("\nQType Frequency:\n", qtype_counts.head())

def main():
    """
    Main function to perform the full EDA on the qa_master dataset.
    """
    # Load the processed dataset
    df = load_data(data_path_processed + 'qa_master.csv')
    
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
    
    # Save the cleaned DataFrame to a new CSV file
    df_cleaned.to_csv(data_path_processed + 'qa_master_processed.csv', index=False)

if __name__ == '__main__':
    main()
