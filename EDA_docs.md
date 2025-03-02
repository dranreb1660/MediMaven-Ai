# **📊 Project Documentation:** 
# *Exploratory Data Analysis (EDA) & ETL Pipeline*

## 📌 Overview
**MediMaven** is an AI-powered medical Q&A chatbot that leverages **Retrieval-Augmented Generation (RAG)** and **Learning-to-Rank (LTR)** techniques to provide high-quality medical answers. The system ingests and processes data from multiple sources, including **MedQuad** and **iCliniq**, ensuring a diverse and reliable knowledge base.

This document details:
- **Exploratory Data Analysis (EDA)** for understanding dataset characteristics.
- **ETL Pipeline Design** for data ingestion, cleaning, transformation, and storage.
- **Visualizations & Insights** to identify patterns, missing data, and distributions.

---

## 📂 Data Overview
### **Sources**
We have combined **two medical Q&A datasets**:
1. **MedQuad**: Curated dataset with structured medical information.
2. **iCliniq**: A dataset of real patient-physician interactions.

### **Columns in the Combined Dataset**
| Column Name   | Description |
|--------------|-------------|
| `Dataset`    | Source dataset (MedQuad / iCliniq) |
| `focus`      | The medical condition or topic covered in the Q&A |
| `synonyms`   | Alternative names for the `focus` |
| `qtype`      | Type of question (e.g., information, treatment, inheritance) |
| `question`   | The user-submitted question |
| `context`    | Additional context for the question |
| `answer`     | Expert or reference response |
| `speciality` | Medical specialty related to the Q&A (iCliniq only) |

### **Key Observations from Initial Data Inspection**
- **MedQuad** has well-defined categories for `focus` and `qtype`, while **iCliniq** has richer `speciality` fields.
- **Missing Data**: `synonyms` and `speciality` have notable gaps.
- **Text Distribution**: Questions tend to be shorter than answers, with an average length of **120 characters** for questions and **500 characters** for answers.

---

# **🔎 EDA Findings from Initial Analysis**

## **Data Snapshot**
- **Total Records**: ~64,798 (approximation from combined MedQuad + iCliniq rows)
- **Columns**: `Dataset`, `focus`, `synonyms`, `qtype`, `question`, `context`, `answer`, and `speciality` (where applicable)

---

## **⚠️ Missing Values**
- **`synonyms` & `speciality`**: Notably higher missing rates, reflecting dataset-specific inconsistencies (e.g., MedQuad may omit synonyms, iCliniq may omit focus).
- **Possible Action**: Apply `fillna` or domain-specific strategies for empty fields (like synonyms or specialties).
![missing values png](data/logs/missing_values.png)


---

## **📊 Dataset Distribution**
- **Balanced Sources**: About 25% came from **MedQuad** while some 75% **iCliniq**
- **Visual Insight**: A bar chart (INSERT) can confirm the proportion.
![missing values png](data/logs/dataset_distribution.png)

---

## **📏 Question & Answer Length**
- **Question Length**: Averages around 120 characters. Short, direct queries are common.
- **Answer Length**: Averages around 500 characters. iCliniq responses often include detailed physician clarifications.
- **Did further** preposing by dropping some rows where the answers had < 10 questions and < 25 words in the context
- **Visualization**: Histograms or box plots highlight distribution and outliers.

![missing values png](data/logs/distribution_of_lengths.png)


---

## **🩺 Top Specialties (iCliniq)**
- **Common Fields**: Obstetrics & Gynecology, General Practitioner, Dermatology, Cardiology, Neurology,
- **Focus**: Shows user interest in broad medical areas.
![missing values png](data/logs/top_specialities.png)

---

## **💡 Text-Based Observations**
- **Frequent Keywords**: "cause", “symptoms,” “treatment,” “pain,” reason,” “complications.”
- **WordCloud (INSERT)** reveals prevalent medical terms across queries.
![missing values png](data/logs/key_words.png)

---

## **🛠 Data Quality Concerns**
- **Duplicates**: Potential overlap if a condition or question is repeated.
- **Empty or Inconsistent Columns**: `synonyms` vs. `speciality` usage.

---

## **🔗 Correlation**
- **Lengths**: Moderate correlation (~0.4–0.5) between question & answer length; longer questions often elicit more detailed answers.
- **Data Type**: Mostly textual, so numeric correlations are less informative beyond length.
![missing values png](data/logs/corelation.png)

---

## 📊 Key Takeaways from EDA
- **Missing Values**: `synonyms` and `speciality` contain gaps, requiring imputation or exclusion strategies.
- **Dataset Balance**: MedQuad and iCliniq contribute a roughly equal share of Q&As.
- **Text Characteristics**: Questions average 120 characters, while answers average 500 characters.
- **Specialties**: Common specializations include **Cardiology, Neurology, and Obstetrics & Gynecology**.
- **Common Medical Terms**: Questions frequently contain terms like *symptoms, treatment, pain, diagnosis*.

---

## 🛠️ ETL Pipeline Design
### **Pseudocode for ETL Workflow**

```plaintext
1. Extract: Load MedQuad and iCliniq datasets
2. Transform:
   - Rename columns for consistency
   - Fill missing values where applicable
   - Add a 'Dataset' column for identification
   - Remove duplicate records
3. Load: Save the cleaned dataset for model training
```
---

## **✅ Next Steps**
- **Data Cleaning**: Address missing columns, unify schema.
- **Feature Engineering**: Potentially add domain-based tags or severity indicators.
- **Model Development**: Leverage cleaned data for fine-tuning a specialized LLM.
- **RAG Approach**: Implement retrieval to fetch relevant context, reduce hallucinations.
- Fine-tune a **domain-specific LLM** using this dataset.
- Implement **retrieval-augmented generation (RAG)** to improve responses.
- Deploy **ETL pipeline using Apache Airflow** for automated data updates.
---


This document serves as a **technical blueprint** for data preprocessing and exploration in MediMaven’s development lifecycle.


