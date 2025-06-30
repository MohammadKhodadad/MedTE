import os
import pandas as pd
from datasets import load_dataset

def create_trec_covid_contrastive_data(output_dir="../data"):
    """
    Loads the BeIR/trec-covid dataset (corpus subset),
    extracts 'title'→sentence1 and 'abstract'→sentence2,
    and writes out a CSV.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the 'corpus' config of the trec-covid dataset
    dataset = load_dataset("BeIR/trec-covid", "corpus")

    # Lists to collect examples
    sentence1 = []   # titles
    sentence2 = []   # abstracts

    # Depending on the HF version, this may be a single split or have 'train'
    splits = dataset.keys() if isinstance(dataset, dict) else ["train"]
    for split in splits:
        ds = dataset[split] if isinstance(dataset, dict) else dataset
        print(f"Processing split: {split}")
        for example in ds:
            title    = example.get("title", "").strip()
            abstract = example.get("text", "").strip()
            # Only include if both fields are non-empty
            if title and abstract:
                sentence1.append(title)
                sentence2.append(abstract)

    # Build DataFrame
    df = pd.DataFrame({
        "sentence1": sentence1,
        "sentence2": sentence2
    })

    # Path to save CSV
    csv_path = os.path.join(output_dir, "trec_covid_contrastive_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    # Example usage
    create_trec_covid_contrastive_data()
