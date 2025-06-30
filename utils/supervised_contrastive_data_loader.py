import os
import torch
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, BertTokenizer

# Step 1: Dataset Class for Supervised Contrastive Learning with Hard Negatives
class SupervisedContrastiveDataset(Dataset):
    def __init__(self, directory_path, tokenizer, max_length=512, num_neg=5):
        self.files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
        data_list = []
        for address in self.files:
            temp_data = pd.read_csv(address).dropna()
            data_list.append(temp_data)
        
        self.data = pd.concat(data_list, axis=0).dropna().reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_neg = num_neg
        
        print(f"Total records: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence1 = self.data.iloc[index]['sentence1']
        sentence2 = self.data.iloc[index]['sentence2']  # Positive sample
        hard_negatives = json.loads(self.data.iloc[index]['hard_negative'])[:self.num_neg]  # Limit to num_neg

        # Concatenate all text inputs
        all_sentences = [sentence1, sentence2] + hard_negatives
        
        # Tokenize all at once
        inputs = self.tokenizer(
            all_sentences, 
            return_tensors="pt", 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        )

        return {
            'input_ids': inputs['input_ids'],  # Shape: (num_neg + 2, seq_len)
            'attention_mask': inputs['attention_mask'],
            'token_type_ids': inputs['token_type_ids']
        }

# Step 2: Collate Function for Contrastive Learning
def collate_func(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])  # Shape: (batch_size, num_neg + 2, seq_len)
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'token_type_ids': token_type_ids
    }

# Step 3: DataLoader Function

def get_supervised_contrastive_dataloader(directory_path, tokenizer, batch_size=16, max_length=512, num_neg=5, distributed=False, rank=0, world_size=1):
    dataset = SupervisedContrastiveDataset(directory_path, tokenizer=tokenizer, max_length=max_length, num_neg=num_neg)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if distributed else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank) if distributed else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not distributed), sampler=train_sampler, collate_fn=collate_func)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, collate_fn=collate_func)
    
    return train_loader, test_loader

# Step 4: Test the DataLoader with BERT
if __name__ == "__main__":
    data = {
        'sentence1': ["This is a positive sentence.", "This is another sentence.","This is a positive sentence.", "This is another sentence."],
        'sentence2': ["This is a similar positive sentence.", "This is a dissimilar sentence.", "This is a similar positive sentence.", "This is a dissimilar sentence."],
        'hard_negative': [json.dumps(["Hard negative 1", "Hard negative 2", "Hard negative 3", "Hard negative 4", "Hard negative 5"]),
                          json.dumps(["Hard negative A", "Hard negative B", "Hard negative C", "Hard negative D", "Hard negative E"]),
                          json.dumps(["Hard negative 1", "Hard negative 2", "Hard negative 3", "Hard negative 4", "Hard negative 5"]),
                          json.dumps(["Hard negative A", "Hard negative B", "Hard negative C", "Hard negative D", "Hard negative E"])]
    }
    df = pd.DataFrame(data)
    df.to_csv("sample_data.csv", index=False)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    train_loader, test_loader = get_supervised_contrastive_dataloader(".", tokenizer, num_neg=5)
    
    for batch in train_loader:
        with torch.no_grad():            
            outputs = model(**{key: batch[key].view(-1, batch['input_ids'].shape[-1]) for key in batch.keys()})
            
        print("Output Shape:", batch.last_hidden_state.shape)
        break
