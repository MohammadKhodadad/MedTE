import os
import requests
import pandas as pd

def create_biorxiv_sentence_data(output_dir="./data", date_range="2016-01-01/2022-12-20"):
    """
    Pages through the bioRxiv API from date_range start→end,
    extracts title→sentence1 and abstract→sentence2,
    and writes a CSV to output_dir/biorxiv_sentences.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    sentence1, sentence2 = [], []
    cursor = 0

    while True:
        url = f"https://api.medrxiv.org/details/biorxiv/{date_range}/{cursor}"
        print(f"Fetching biorxiv: {url}")
        resp = requests.get(url); resp.raise_for_status()
        data = resp.json().get("collection", [])
        if not data:
            break

        for item in data:
            title   = item.get("title", "").strip()
            abstract= item.get("abstract", "").strip()
            if title and abstract:
                sentence1.append(title)
                sentence2.append(abstract)

        cursor += len(data)

    df = pd.DataFrame({"sentence1": sentence1, "sentence2": sentence2})
    out_path = os.path.join(output_dir, "biorxiv_sentences.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} biorxiv rows to {out_path}")


def create_medrxiv_sentence_data(output_dir="./data", date_range="2016-01-01/2024-12-31"):
    """
    Pages through the medRxiv API from date_range start→end,
    extracts title→sentence1 and abstract→sentence2,
    and writes a CSV to output_dir/medrxiv_sentences.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    sentence1, sentence2 = [], []
    cursor = 0

    while True:
        url = f"https://api.medrxiv.org/details/medrxiv/{date_range}/{cursor}"
        print(f"Fetching medrxiv: {url}")
        resp = requests.get(url); resp.raise_for_status()
        data = resp.json().get("collection", [])
        if not data:
            break

        for item in data:
            title   = item.get("title", "").strip()
            abstract= item.get("abstract", "").strip()
            if title and abstract:
                sentence1.append(title)
                sentence2.append(abstract)

        cursor += len(data)

    df = pd.DataFrame({"sentence1": sentence1, "sentence2": sentence2})
    out_path = os.path.join(output_dir, "medrxiv_sentences.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} medrxiv rows to {out_path}")


if __name__ == "__main__":
    # will produce:
    #   ./data/biorxiv_sentences.csv
    #   ./data/medrxiv_sentences.csv
    create_biorxiv_sentence_data()
    create_medrxiv_sentence_data()
