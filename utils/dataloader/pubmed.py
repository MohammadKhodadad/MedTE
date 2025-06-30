import os
import time
import tqdm
import pandas as pd
from Bio import Entrez

# Set your email (required by PubMed API)
Entrez.email = "your_email@example.com"

def pubmed_get_article_details_batch(pmids, batch_size=1000):
    """
    Fetch article details (title + abstract) for a batch of PMIDs from PubMed.

    :param pmids: List of PubMed IDs.
    :param batch_size: Number of articles to retrieve per batch.
    :return: List of dictionaries containing titles and abstracts.
    """
    abstracts = []
    
    for i in tqdm.tqdm(tqdm.tqdm(range(0, len(pmids), batch_size))):
        batch_pmids = pmids[i:i+batch_size]
        pmid_str = ",".join(batch_pmids)
        
        try:
            handle = Entrez.efetch(db='pubmed', id=pmid_str, retmode='xml')
            xml_data = Entrez.read(handle)
            handle.close()
            
            for article in xml_data.get('PubmedArticle', []):
                title = article['MedlineCitation']['Article'].get('ArticleTitle', 'N/A')
                abstract_data = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', 'N/A')
                abstract = ' '.join(abstract_data) if isinstance(abstract_data, list) else abstract_data
                
                abstracts.append({
                    "Title": title,
                    "Abstract": abstract,
                })
        
        except Exception as e:
            print(f"Error fetching batch {i//batch_size + 1}: {e}")
        
        time.sleep(0.25)  # Avoid hitting PubMed rate limits

    return abstracts

def fetch_pubmed_abstracts(query, max_results_per_month, years):
    """
    Fetch abstracts from PubMed in batch mode for improved efficiency.

    :param query: PubMed search query.
    :param max_results_per_month: Maximum results to fetch per month.
    :param years: List of years to fetch data for.
    :return: List of article details (title + abstract).
    """
    id_list = set()  # Use a set to store unique IDs

    for year in years:
        for month in range(1, 13):
            start_date = f"{year}/{month:02d}/01"
            end_date = f"{year}/{month:02d}/15"
            query_with_date = f"{query} AND ({start_date}[PDAT] : {end_date}[PDAT])"

            print(f"Fetching data for: {query_with_date}")

            try:
                handle = Entrez.esearch(db="pubmed", term=query_with_date, retmax=max_results_per_month)
                record = Entrez.read(handle)
                handle.close()

                monthly_ids = set(record.get("IdList", []))  # Convert to set to remove duplicates
                id_list.update(monthly_ids)

                print(f"Year {year}, Month {month}: Found {len(monthly_ids)} articles.")
            except Exception as e:
                print(f"Error fetching article IDs for {query_with_date}: {e}")

            time.sleep(0.25)  # Avoid hitting PubMed rate limits

            start_date = f"{year}/{month:02d}/16"
            end_date = f"{year}/{month:02d}/30"
            query_with_date = f"{query} AND ({start_date}[PDAT] : {end_date}[PDAT])"

            print(f"Fetching data for: {query_with_date}")

            try:
                handle = Entrez.esearch(db="pubmed", term=query_with_date, retmax=max_results_per_month)
                record = Entrez.read(handle)
                handle.close()

                monthly_ids = set(record.get("IdList", []))  # Convert to set to remove duplicates
                id_list.update(monthly_ids)

                print(f"Year {year}, Month {month}: Found {len(monthly_ids)} articles.")
            except Exception as e:
                print(f"Error fetching article IDs for {query_with_date}: {e}")

            time.sleep(0.25)  # Avoid hitting PubMed rate limits

    if not id_list:
        print("No articles found for the query.")
        return []

    print(f"Total collected unique articles across years: {len(id_list)}")

    # Fetch article details in batches
    abstracts = pubmed_get_article_details_batch(list(id_list))

    return abstracts

def save_abstracts_for_mlm(abstracts, output_dir="data", name="no_name"):
    """
    Save abstracts stacked together for MLM training.

    :param abstracts: List of abstracts.
    :param output_dir: Directory to save the abstracts.
    :param name: Name to use in the output filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"pubmed_mlm_{name}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for abstract in abstracts:
            if abstract["Abstract"] != "N/A":
                f.write(abstract["Abstract"] + "\n")

    print(f"MLM corpus saved to {output_file}")

def save_abstracts_for_contrastive(abstracts, output_dir="data", name="no_name"):
    """
    Save title and abstracts for contrastive learning.

    :param abstracts: List of abstracts.
    :param output_dir: Directory to save the contrastive learning data.
    :param name: Name to use in the output filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"pubmed_cl_{name}.csv")
    df = pd.DataFrame(abstracts)
    df = df[df["Title"].notna() & df["Abstract"].notna()]  # Filter out rows with missing data
    df = df.rename(columns={'Title': 'sentence1', 'Abstract': 'sentence2'})
    df.to_csv(output_file, index=False, columns=["sentence1", "sentence2"], encoding="utf-8")

    print(f"Contrastive learning data saved to {output_file}")

def download_pubmed_mlm(output_dir, max_results=50000, query='("Therapeutics"[MeSH Terms] OR "Epidemiology"[MeSH Terms] OR "Pathology"[MeSH Terms])'):
    years = [2023, 2024]
    pubmed_abstracts = fetch_pubmed_abstracts(query, max_results, years)
    save_abstracts_for_mlm(pubmed_abstracts, output_dir=output_dir, name=query.replace(' ', '_'))

def download_pubmed_cl(output_dir, max_results=500000, query='("Therapeutics"[MeSH] OR Pathology OR Clinical Trials OR medicine OR "Medicine"[MeSH] OR "Epidemiology"[MeSH] OR "Pathology"[MeSH] OR "Diagnosis"[MeSH] OR "Treatment Outcome"[MeSH] OR "Clinical Medicine"[MeSH] OR "Public Health"[MeSH] OR "Pharmacology"[MeSH] OR "Internal Medicine"[MeSH] OR "Surgery"[MeSH] OR "Genetics"[MeSH] OR "Molecular Biology"[MeSH] OR "Medical Informatics"[MeSH] OR "Precision Medicine"[MeSH] OR "Evidence-Based Medicine"[MeSH])'):
    years = [2024,2023,2022]
    pubmed_abstracts = fetch_pubmed_abstracts(query, max_results, years)
    save_abstracts_for_contrastive(pubmed_abstracts, output_dir=output_dir, name='pubmed_pairs2')

if __name__ == "__main__":
    # Example query
    download_pubmed_cl('.')
