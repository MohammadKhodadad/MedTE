import os
import tqdm
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
import torch
import pickle
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import random_split
from nltk.tokenize import sent_tokenize
# Step 1: Load and tokenize the text files
class TokenizedChunkedDataset(Dataset):
    def __init__(self, directory_path, tokenizer, chunk_size=512):
        # Get all the files from the directory
        self.files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

        # Step 1: Load and concatenate all text files
        self.full_text = self.load_all_text_files()
        # print(f"full_text: {len(self.full_text)}")
        # self.full_text = self.full_text [:2000000]
        print(f"full_text: {len(self.full_text)}")
        # Step 2: Tokenize the full concatenated text without truncation
        self.tokenized_data = self.tokenize_full_text()
        print(f"tokenized_data: {len(self.tokenized_data)}")
        # Step 3: Calculate the number of chunks based on the chunk size
        self.num_chunks = len(self.tokenized_data)
        print(f"THE NUMBER OF CHUNKS: {self.num_chunks}")
    def load_all_text_files(self):
        full_text = ""
        print("Loaded Files")
        print(self.files)
        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                full_text += f.read() + " "  # Concatenate all the text files into a single text
            print(len(full_text))
        return full_text

    def tokenize_full_text(self):
        # Step 1: Clean up the text to handle newline characters
        cleaned_text = self.full_text.replace('\n', ' ')  # Replace newlines with a space

        # Step 2: Split the cleaned text into sentences
        sentences = sent_tokenize(cleaned_text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in tqdm.tqdm(sentences):
            # Tokenize the current sentence without truncation
            tokenized_sentence = self.tokenizer(sentence, return_tensors="pt", padding=False, truncation=False) #IMPORTANT# todo: Don't add special tokens.
            sentence_length = len(tokenized_sentence['input_ids'][0])

            # Check if adding this sentence would exceed the chunk size
            if current_length + sentence_length > self.chunk_size:
                # If yes, tokenize the accumulated chunk and store it
                concatenated_text = " ".join(current_chunk)
                tokenized_chunk = self.tokenizer(
                    concatenated_text,
                    return_tensors="pt",
                    padding='max_length',
                    max_length=self.chunk_size,
                    truncation=True
                )
                chunks.append(tokenized_chunk)
                current_chunk = []
                current_length = 0

            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_length += sentence_length
        return chunks
    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        # Retrieve the tokenized chunk at the specified index
        tokenized_chunk = self.tokenized_data[idx]

        # Extract input_ids and attention_mask from the tokenized chunk
        input_ids_chunk = tokenized_chunk['input_ids'].squeeze()
        attention_mask_chunk = tokenized_chunk['attention_mask'].squeeze()

        # Handle token_type_ids if present
        token_type_ids_chunk = tokenized_chunk.get('token_type_ids')
        if token_type_ids_chunk is not None:
            token_type_ids_chunk = token_type_ids_chunk.squeeze()
        else:
            token_type_ids_chunk = torch.zeros_like(input_ids_chunk)

        return {
            'input_ids': input_ids_chunk,
            'attention_mask': attention_mask_chunk,
            'token_type_ids': token_type_ids_chunk
        }


# Step 2: DataLoader function for efficient retrieval
# def get_mlm_dataloader(directory_path, tokenizer, batch_size=32, max_length=512,mlm_probability=0.15):
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
#     )
#     dataset = TokenizedChunkedDataset(directory_path, tokenizer=tokenizer, chunk_size=max_length)
#     train_size = int(0.8 * len(dataset))  # 80% for training
#     test_size = len(dataset) - train_size  # 20% for testing

#     # Split the dataset
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#     # Create DataLoaders for train and test sets
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
#     return train_loader,test_loader


def get_mlm_dataloader(directory_path, tokenizer, batch_size=32, max_length=512, mlm_probability=0.15, distributed=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    dataset = TokenizedChunkedDataset(directory_path, tokenizer=tokenizer, chunk_size=max_length)
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print('number of records:',len(dataset))
    # Use DistributedSampler if in distributed mode
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    test_sampler = DistributedSampler(test_dataset) if distributed else None

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not distributed), sampler=train_sampler, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, collate_fn=data_collator)
    print(f'number of batches:{len(train_loader)}x{batch_size} or {len(train_dataset)*batch_size} records', flush=True)
    return train_loader, test_loader


# Step 3: Example usage
if __name__ == "__main__":
    # Directory containing .txt files
    directory_path = "../data"
    
    # Load the tokenizer
    from tokenizer_loader import load_tokenizer
    tokenizer=load_tokenizer()
    # Load the DataLoader
    tokenized_dataloader = get_mlm_dataloader(directory_path, tokenizer)

    # Iterate through the DataLoader and inspect the batches
    for batch in tokenized_dataloader:
        print("Input IDs:", batch['input_ids'][0])
        print(tokenizer.decode(batch['input_ids'][0]))
        print("Input IDs:", batch['labels'][0])
        print(tokenizer.decode(batch['labels'][0][batch['labels'][0]>=0]))
        # print("Attention Mask:", batch['attention_mask'][0])
        # print("Token type ids:", batch['token_type_ids'][0])
        
