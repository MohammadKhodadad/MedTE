import os
import pandas as pd
from datasets import load_dataset

def create_curev1_contrastive_learning_data(output_dir="../data"):
    """
    Loads the clinia/CUREv1 dataset, concatenates all splits,
    renames 'title' to 'sentence1' and 'text' to 'sentence2',
    and writes out a CSV.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the CUREv1 dataset from Hugging Face
    dataset = load_dataset("clinia/CUREv1")

    # Lists to collect examples
    sentence1 = []   # titles
    sentence2 = []   # texts

    # Iterate over each split in the dataset
    for split in dataset:
        sentence1 = []   # titles
        sentence2 = []
        print(f"Processing split: {split}")
        for example in dataset[split]:
            title = example.get("title", "").strip()
            text  = example.get("text", "").strip()

            # Only include if both fields are non-empty
            if title and text:
                sentence1.append(title)
                sentence2.append(text)

        # Build DataFrame
        df = pd.DataFrame({
            "sentence1": sentence1,
            "sentence2": sentence2
        })

        # Path to save CSV
        csv_path = os.path.join(output_dir, f"curev1_{split}_contrastive_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to {csv_path}")

if __name__ == "__main__":
    create_curev1_contrastive_learning_data()
