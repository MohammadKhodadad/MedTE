import os
import pandas as pd
from datasets import load_dataset

def create_medmcqa_contrastive_leanring_data(output_dir="../data"):
    # Ensure the output directory exists
    # Load the MedMCQA dataset from Hugging Face
    dataset = load_dataset("openlifescienceai/medmcqa")

    # Initialize lists for retrieval
    queries = []      # Questions
    documents = []    # Explanations

    # Iterate over each split in the dataset (train, validation, test)
    for split in dataset:
        print(f"Processing split: {split}")
        for example in dataset[split]:
            question = example.get("question", "")
            explanation = example.get("exp", "")

            # Ensure both question and explanation are not empty
            if question and explanation:
                queries.append(question)
                documents.append(explanation)

    # Create a DataFrame with the retrieval data
    df = pd.DataFrame({
        "sentence1": queries,
        "sentence2": documents
    })

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(output_dir, "contrastive_medqmcqa_data.csv"), index=False)
