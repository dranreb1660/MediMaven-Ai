
import xml.etree.ElementTree as ET
import pandas as pd
import glob


def extract_one_medquad_file(xml_file):
    """
    Parses a single MedQuAD XML where <Document> is the root element.
    Extracts synonyms, semantic group, question, and answer.

    Args:
        xml_file (str): Path to the MedQuAD XML file.

    Returns:
        List[dict]: A list of Q&A dictionaries containing relevant fields.
    """

    # Parse the XML
    tree = ET.parse(xml_file)
    root = tree.getroot()  # <Document> is the root in your snippet

    # Basic Document-level attributes
    doc_id = root.get("id", "")
    source = root.get("source", "")
    url = root.get("url", "")

    # Focus: Main topic/disease name
    focus = root.findtext("Focus", default="")

    # Extract synonyms
    synonyms = []
    syn_path = root.find("FocusAnnotations/Synonyms")
    if syn_path is not None:
        for syn_el in syn_path.findall("Synonym"):
            if syn_el.text:
                synonyms.append(syn_el.text.strip())

    # Extract semantic group
    sem_group = None
    sem_group_el = root.find("FocusAnnotations/UMLS/SemanticGroup")
    if sem_group_el is not None and sem_group_el.text:
        sem_group = sem_group_el.text.strip()

    # Collect Q&A pairs
    qa_data = []
    qapairs = root.find("QAPairs")
    if qapairs is not None:
        for pair in qapairs.findall("QAPair"):
            pid = pair.get("pid", "")
            question_el = pair.find("Question")
            answer_el = pair.find("Answer")

            question = question_el.text if question_el is not None else ""
            qtype = question_el.get("qtype") if question_el is not None else ""
            answer = answer_el.text if answer_el is not None else ""

            question =  question and question.strip()
            answer = answer  and answer.strip()

            # Build a record for each Q&A
            qa_data.append({
                "document_id": doc_id,
                "source": source,
                "url": url,
                "focus": focus,
                "synonyms": ','.join(synonyms) if synonyms else '',
                "semantic_group": sem_group,
                "pid": pid,
                "qtype": qtype,
                "question": question,
                "answer": answer
            })


    return qa_data



def extract_medquad(xml_files):
    # Initialize storage lists
    all_records = []
    for file in xml_files:
        one_record = extract_one_medquad_file(file)
        all_records.extend(one_record)

    df = pd.DataFrame(all_records)

    return df

all_xml_files = glob.glob('./data/raw/medquad/*/*.xml', recursive=True)
print(len(all_xml_files))

if __name__ == '__main__':
    csv_out = "./data/processed/medquad_extracted.csv"
    df_medquad = extract_medquad(all_xml_files)
    print(df_medquad)
    df_medquad.to_csv(csv_out, index=False)
    print(f"CSV saved to {csv_out}")