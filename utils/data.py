import os
import re
import tqdm
import pandas as pd
from Bio import Entrez
import os
import time

def clean_text(text):
    # Lowercase the text
    text = text.lower() 
    # Remove special characters (except necessary punctuation like periods, commas)
    text = re.sub(r'[^a-z0-9\s.,]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n+', ' ', text) #REPLACED \n with ' '
    return text

def create_txt_from_csv(csv_address, target_directory):
    # Read the CSV file
    base_name = os.path.basename(csv_address).replace('.csv', '')
    df = pd.read_csv(csv_address)
    # Select the required columns
    selected_columns = ['history_of_present_illness', 'chief_complaint', 'discharge_diagnosis']
    df_selected = df[selected_columns]
    print(f"shape before dropping nones: {df_selected.shape}")
    df_selected=df_selected.dropna()
    print(f"shape after dropping nones: {df_selected.shape}")
    # Check if the target directory exists, create if not
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Iterate over the rows, concatenate columns, clean the text, and create text files for each row
    print("PREPARING DATA FOR MLM....")
    file_path = os.path.join(target_directory, f"{base_name}_cleaned.txt")
    with open(file_path, 'w') as file:
        for idx, row in tqdm.tqdm(df_selected.iterrows()):
            concatenated_text = ' '.join(row.values.astype(str))
            cleaned_text = clean_text(concatenated_text)
            file.write(cleaned_text+' ') #REPLACED \n with ' '

    return f"Text files created in {target_directory}"


def create_cl_data_from_csv(csv_address, target_directory, col1, col2):
    # Read the CSV file
    base_name = os.path.basename(csv_address).replace('.csv', '')
    df = pd.read_csv(csv_address)
    if type(col1)==list:
        col=df[col1[0]]
        for i in range(1,len(col1)):
            col=col+df[col1[i]]
        df["_and_".join(col1)]=col
        col1 = "_and_".join(col1)
    
    if type(col2)==list:
        col=df[col2[0]]
        for i in range(1,len(col2)):
            col=col+df[col2[i]]
        df["_and_".join(col2)]=col
        col2 = "_and_".join(col2)
    # Select the two columns provided for contrastive learning
    df_selected = df[[col1, col2]]
    print(f"shape before dropping nones: {df_selected.shape}")
    df_selected=df_selected.dropna()
    print(f"shape after dropping nones: {df_selected.shape}")
    # Rename the columns to text1 and text2
    df_selected.columns = ['sentence1', 'sentence2']
    
    # Clean the text in both columns
    print("PREPARING DATA FOR CONTRASTIVE LEARNING....")
    df_selected['sentence1'] = df_selected['sentence1'].apply(clean_text)
    df_selected['sentence2'] = df_selected['sentence2'].apply(clean_text)
    
    # Save the new DataFrame as a CSV file
    output_csv_path = os.path.join(target_directory, f'{col1}_vs_{col2}_cleaned.csv')
    df_selected.to_csv(output_csv_path, index=False)


# WORK HERE

def pubmed_get_article_details(pmid):

    handle = Entrez.efetch(db='pubmed', id=str(pmid), retmode='xml')
    xml_data = Entrez.read(handle)
    handle.close()

    article_data = xml_data['PubmedArticle'][0]
    title = article_data['MedlineCitation']['Article'].get('ArticleTitle', '')
    abstract_data = article_data['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', '')
    abstract = ''.join(abstract_data) if abstract_data else ''

    return {
        "Title": title,
        "Abstract": abstract,
    }

def fetch_pubmed_abstracts(query, email, max_results=100, year=2024):

    Entrez.email = email

    # Search PubMed for the query
    query_with_year = f"{query} AND {year}[DP]"
    handle = Entrez.esearch(db="pubmed", term=query_with_year, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()

    # Fetch article details using IDs
    id_list = record["IdList"]
    if not id_list:
        print("No articles found for the query.")
        return []

    print(f"Found {len(id_list)} articles. Fetching abstracts...")

    abstracts = []
    for pmid in id_list:
        try:
            article_details = pubmed_get_article_details(pmid)
            abstracts.append(article_details)
        except Exception as e:
            print(f"Error fetching data for PMID {pmid}: {e}")

    return abstracts


def save_abstracts_for_mlm(abstracts, output_dir="data",name="no_name"):
    """
    Save abstracts stacked together for MLM training.

    :param abstracts: List of abstracts.
    :param output_dir: Directory to save the abstracts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"pubmed_mlm_{name}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for abstract in abstracts:
            if abstract["Abstract"] != "N/A":
                f.write(abstract["Abstract"] + "\n")

    print(f"MLM corpus saved to {output_file}")

def save_abstracts_for_contrastive(abstracts, output_dir="data",name="no_name"):
    """
    Save title and abstracts for contrastive learning.

    :param abstracts: List of abstracts.
    :param output_dir: Directory to save the contrastive learning data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"pubmed_cl_{name}.csv")
    df = pd.DataFrame(abstracts)
    df = df[df["Title"].notna() & df["Abstract"].notna()]  # Filter out rows with missing data
    df.to_csv(output_file, index=False, columns=["Title", "Abstract"], encoding="utf-8")

    print(f"Contrastive learning data saved to {output_file}")

if __name__ == "__main__":
    # Example query
    query = "Cancer"
    email = "your_email@example.com"  # Replace with your email
    max_results = 2
    year = 2024

    pubmed_abstracts = fetch_pubmed_abstracts(query, email, max_results, year)
    save_abstracts_for_mlm(pubmed_abstracts,name=query.replace(' ','_'))
    save_abstracts_for_contrastive(pubmed_abstracts,name=query.replace(' ','_'))
