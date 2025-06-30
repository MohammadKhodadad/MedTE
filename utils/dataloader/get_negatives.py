import os
import json
import pandas as pd
import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm

def load_csv_files(directory):
    """Loads all CSV files in the directory and returns a combined DataFrame."""
    all_data = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            df = df.dropna()
            if {'sentence1', 'sentence2'}.issubset(df.columns):
                all_data.append(df) 
    return pd.concat(all_data, ignore_index=True) if all_data else None

def mean_pooling(model_output, attention_mask):
    """Applies mean pooling to get sentence embeddings."""
    token_embeddings = model_output[0]  # First element has all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_sentences(sentences, model, tokenizer, batch_size=64):
    """Encodes sentences using the transformer model and returns embeddings."""
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding Sentences"):
            try:
                batch = sentences[i:i+batch_size]
                tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
                model_output = model(**tokens)
                pooled_output = mean_pooling(model_output, tokens['attention_mask'])
                normalized_emb = F.normalize(pooled_output, p=2, dim=1)
                embeddings.append(normalized_emb.cpu().numpy())
            except Exception as e:
                print(e)
                print(batch)
    return np.vstack(embeddings)

def save_embeddings(embeddings, file_path):
    """Saves embeddings as a NumPy file."""
    np.save(file_path, embeddings)
    print(f"Embeddings saved to {file_path}")

def load_embeddings(file_path):
    """Loads embeddings from a NumPy file."""
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print(f"File {file_path} not found!")
        return None

def find_hard_negatives(query_embs, doc_embs, k=5):
    """Finds hard negatives using FAISS nearest neighbor search."""
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs.astype(np.float32))
    scores, indices = index.search(query_embs.astype(np.float32), k+1)  # +1 to avoid self-matches
    hard_negatives = [indices[i][1:] for i in range(len(indices))]  # Skip the first (self-match)
    return hard_negatives

def save_hard_negatives(directory, model_name="thenlper/gte-base", embedding_file="embeddings.npy", output_file="processed_data.csv"):
    print("Loading data...")
    df = load_csv_files(directory)
    if df is None:
        print("No valid CSV files found!")
        return
    
    sentences = list(df['sentence1']) + list(df['sentence2'])
    
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Embedding sentences...")
    embeddings = embed_sentences(sentences, model, tokenizer)
    save_embeddings(embeddings, embedding_file)
    print("Embeddings saved.")

def load_and_process_hard_negatives(directory, embedding_file, output_file):
    print("Loading data...")
    df = load_csv_files(directory)
    if df is None:
        print("No valid CSV files found!")
        return
    
    print("Loading precomputed embeddings...")
    embeddings = load_embeddings(embedding_file)
    if embeddings is None:
        return
    
    print("Finding hard negatives...")
    hard_negatives = find_hard_negatives(embeddings[len(df):], embeddings[len(df):], k=5)
    
    df['hard_negative'] = [json.dumps([df.iloc[id_]['sentence2'] for id_ in idx]) for idx in hard_negatives]
    
    print("Saving processed data...")
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    directory = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/csvs"  # Change this to your CSV directory
    embedding_file = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/embeddings.npy"
    output_file = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/hard_neg/hard_negative_v1.csv"
    
    save_hard_negatives(directory, embedding_file=embedding_file)
    load_and_process_hard_negatives(directory, embedding_file=embedding_file, output_file=output_file)
