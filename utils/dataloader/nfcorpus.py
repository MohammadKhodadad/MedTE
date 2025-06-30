import os
import pandas as pd
from datasets import load_dataset

def create_nfcorpus_contrastive_learning_data(output_dir="../data"):
    """
    Loads the BeIR/nfcorpus dataset (corpus subset),
    extracts 'title'→sentence1 and 'text'→sentence2,
    and writes out a CSV.
    """
    # 1) Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2) Load the 'corpus' config of nfcorpus
    dataset = load_dataset("BeIR/nfcorpus", "corpus")

    # 3) Prepare lists
    sentence1, sentence2 = [], []

    # 4) Handle single- vs multi-split returns
    if isinstance(dataset, dict):
        splits = dataset.keys()
    else:
        splits = ["train"]
        dataset = {"train": dataset}

    # 5) Iterate and collect
    for split in splits:
        print(f"Processing split: {split}")
        for example in dataset[split]:
            title = example.get("title", "").strip()
            text  = example.get("text",  "").strip()
            if title and text:
                sentence1.append(title)
                sentence2.append(text)

    # 6) Build DataFrame
    df = pd.DataFrame({
        "sentence1": sentence1,
        "sentence2": sentence2
    })

    # 7) Save CSV
    csv_path = os.path.join(output_dir, "nfcorpus_contrastive_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    create_nfcorpus_contrastive_learning_data()
