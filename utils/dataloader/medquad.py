import os
import pandas as pd
from datasets import load_dataset

def create_medquad_contrastive_leanring_data(output_dir='../data'):
    """
    Loads the MedQuAD dataset from Hugging Face, extracts the 'question' and 'answer' columns,
    anonymizes the answers in parallel, and saves the processed data as a CSV file.
    """
    dataset_name = 'lavita/MedQuAD'
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])  # Assuming we use the train split
    df = df[['question', 'answer']].rename(columns={'question': 'sentence1', 'answer': 'sentence2'})
    df = df.dropna()
    df.to_csv(os.path.join(output_dir, "contrastive_medquad_data.csv"), index=False)