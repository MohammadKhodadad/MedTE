import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


from torch.utils.data import random_split
# Step 2: Dataset Class for Contrastive Learning
class ContrastiveDataset(Dataset):
    def __init__(self, directory_path, tokenizer, max_length=512):
        self.files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
        data_list=[]
        for address in self.files:
            temp_data=pd.read_csv(address)
            temp_data= temp_data.dropna()
            print(temp_data.shape)
            data_list.append(temp_data)

        self.data = pd.concat(data_list,axis=0)
        print(self.data.shape)
        self.data=self.data.dropna().reset_index(drop=True)
        print(self.data.shape)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"THE NUMBER OF RECORDS: {len(self.data)}")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the sentence pair
        sentence1 = self.data.iloc[index]['sentence1']
        sentence2 = self.data.iloc[index]['sentence2']

        # Tokenize the sentence pair
        # try:
        inputs1 = self.tokenizer(sentence1, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True)
        inputs2 = self.tokenizer(sentence2, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True)
        # except:
        #     print(sentence1)
        #     print(sentence2)
        #     raise Exception('error')
        # Return the tokenized inputs for contrastive learning
        return {
            'input_ids1': inputs1['input_ids'].squeeze(),  # Squeeze to remove extra dimension
            'attention_mask1': inputs1['attention_mask'].squeeze(),
            'token_type_ids1': inputs1['token_type_ids'].squeeze(),
            'input_ids2': inputs2['input_ids'].squeeze(),
            'attention_mask2': inputs2['attention_mask'].squeeze(),
            'token_type_ids2': inputs2['token_type_ids'].squeeze(),
        }

def collate_func(batch):
    # Separate out input_ids1, attention_mask1, token_type_ids1
    input_ids1 = torch.stack([item['input_ids1'] for item in batch])
    attention_mask1 = torch.stack([item['attention_mask1'] for item in batch])
    token_type_ids1 = torch.stack([item['token_type_ids1'] for item in batch])
    
    # Create batch1 dictionary
    batch1 = {
        'input_ids': input_ids1,
        'attention_mask': attention_mask1,
        'token_type_ids': token_type_ids1
    }

    # Separate out input_ids2, attention_mask2, token_type_ids2 (move them to device if necessary)
    input_ids2 = torch.stack([item['input_ids2'] for item in batch])
    attention_mask2 = torch.stack([item['attention_mask2'] for item in batch])
    token_type_ids2 = torch.stack([item['token_type_ids2'] for item in batch])
    
    # Create batch2 dictionary
    batch2 = {
        'input_ids': input_ids2,
        'attention_mask': attention_mask2,
        'token_type_ids': token_type_ids2
    }
    return batch1, batch2

# Step 3: DataLoader Function
# def get_contrastive_dataloader(dataframe,tokenizer, batch_size=32, max_length=512):
#     dataset = ContrastiveDataset(dataframe, tokenizer=tokenizer, max_length=max_length)
#     train_size = int(0.8 * len(dataset))  # 80% for training
#     test_size = len(dataset) - train_size  # 20% for testing

#     # Split the dataset
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#     # Create DataLoaders for train and test sets
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_func)

#     return train_loader, test_loader

def get_contrastive_dataloader(dataframe,tokenizer, batch_size=32, max_length=512, distributed=False, rank=0, world_size=1):
    dataset = ContrastiveDataset(dataframe, tokenizer=tokenizer, max_length=max_length)
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Use DistributedSampler if in distributed mode
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if distributed else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank) if distributed else None

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not distributed), sampler=train_sampler, collate_fn=collate_func)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, collate_fn=collate_func)

    return train_loader, test_loader

# Step 4: Test the DataLoader
if __name__ == "__main__":
    # Get the DataLoader
    # Step 1: Sample DataFrame (simulating the 2-column CSV file)
    data = {'sentence1': ["This is a positive sentence.", "This is another sentence."],
            'sentence2': ["This is a similar positive sentence.", "This is a dissimilar sentence."]}
    df = pd.DataFrame(data)
    from tokenizer_loader import load_tokenizer
    tokenizer=load_tokenizer()

    contrastive_dataloader = get_contrastive_dataloader(df,tokenizer)

    # Iterate through the DataLoader
    for batch1, batch2 in contrastive_dataloader:
        print("Input IDs Sentence 1:", batch1['input_ids'])
        print("Attention Mask Sentence 1:", batch1['attention_mask'])
        print("Input IDs Sentence 2:", batch2['input_ids'])
        print("Attention Mask Sentence 2:", batch2['attention_mask'])
