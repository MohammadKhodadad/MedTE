import os
import pandas as pd
from datasets import load_dataset
def create_medqa_contrastive_leanring_data(output_dir="../data"):
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    sentence1 = []  # Questions
    sentence2 = []  # Options
    labels = []     # 1 for correct, 0 for incorrect

    # Iterate over each split in the dataset (train, test)
    for split in dataset:
        print(f"Processing split: {split}")
        for example in dataset[split]:
            question = example.get("question", None)
            correct_answer_idx = example.get("answer_idx", None)  # Index of the correct answer (0-based)
            options = example.get("options", None)

            # Create pairs for each option
            if correct_answer_idx and options and question:
                sentence1.append(question)
                sentence2.append(options[correct_answer_idx])

    # Create a DataFrame with the pair classification data
    df = pd.DataFrame({
        "sentence1": sentence1,
        "sentence2": sentence2,
    })

    df=df.dropna()
    df.to_csv(os.path.join(output_dir, "contrastive_medqa_data.csv"), index=False)
    
