import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
import random

# Dataset that splits each data source into train and test parts.
class PerSourceContrastiveDataset(Dataset):
    def __init__(self, data_source, tokenizer, split='train', train_frac=0.80, max_length=512, initial_shuffle=True, shuffle_seed=None):
        """
        data_source: either a directory path (str) with CSV files or a single pandas DataFrame.
        tokenizer: tokenizer for processing sentences.
        split: 'train' or 'test' to indicate which split to use.
        train_frac: fraction of rows (per source) to use for training.
        max_length: maximum token length for tokenization.
        initial_shuffle: whether to shuffle rows within each source when loading.
        shuffle_seed: seed for reproducibility.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split.lower()
        if self.split not in ['train', 'test']:
            raise ValueError("split must be 'train' or 'test'")

        # Load each CSV (or use the provided DataFrame) as a separate data source.
        if isinstance(data_source, str):
            files = [os.path.join(data_source, f) for f in os.listdir(data_source) if f.endswith('.csv')]
            sources = []
            for file in files:
                
                print(f'Processing address: {file}')
                try:
                    df = pd.read_csv(file).dropna().drop_duplicates(subset=['sentence1']).reset_index(drop=True)
                    if initial_shuffle:
                        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
                    sources.append(df)
                except Exception as e:
                    print(e)
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.drop_duplicates(subset=['sentence1']).dropna().reset_index(drop=True)
            if initial_shuffle:
                df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
            sources = [df]
        elif isinstance(data_source, list):
            sources = []
            for data_item in data_source:
                df = data_item.drop_duplicates(subset=['sentence1']).dropna().reset_index(drop=True)
                if initial_shuffle:
                    df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
                sources.append(df)
        else:
            raise ValueError("data_source must be a directory path (str) or a pandas DataFrame.")

        # For each source, split into train and test parts.
        self.dataframes = []
        for df in sources:
            n = len(df)
            split_idx = int(n * train_frac)
            if self.split == 'train':
                self.dataframes.append(df.iloc[:split_idx].reset_index(drop=True))
            else:
                self.dataframes.append(df.iloc[split_idx:].reset_index(drop=True))
        
        self._compute_cum_lengths()
        print(f"{self.split.capitalize()} dataset: {len(self.dataframes)} source(s), total records = {len(self)}")

    def _compute_cum_lengths(self):
        self.cum_lengths = []
        total = 0
        for df in self.dataframes:
            total += len(df)
            self.cum_lengths.append(total)

    def reshuffle(self, shuffle_seed=None):
        """Reshuffle each data source independently."""
        for i in range(len(self.dataframes)):
            self.dataframes[i] = self.dataframes[i].sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        self._compute_cum_lengths()

    def __len__(self):
        return self.cum_lengths[-1] if self.cum_lengths else 0

    def __getitem__(self, index):
        # Identify which data source the index belongs to.
        source_idx = 0
        for cum_length in self.cum_lengths:
            if index < cum_length:
                break
            source_idx += 1
        # Determine row index within that data source.
        row_idx = index if source_idx == 0 else index - self.cum_lengths[source_idx - 1]
        row = self.dataframes[source_idx].iloc[row_idx]
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']

        # Tokenize the sentences.
        inputs1 = self.tokenizer(sentence1, return_tensors="pt", max_length=self.max_length,
                                 padding='max_length', truncation=True)
        inputs2 = self.tokenizer(sentence2, return_tensors="pt", max_length=self.max_length,
                                 padding='max_length', truncation=True)

        return {
            'input_ids1': inputs1['input_ids'].squeeze(),
            'attention_mask1': inputs1['attention_mask'].squeeze(),
            'token_type_ids1': inputs1.get('token_type_ids', torch.zeros_like(inputs1['input_ids'])).squeeze(),
            'input_ids2': inputs2['input_ids'].squeeze(),
            'attention_mask2': inputs2['attention_mask'].squeeze(),
            'token_type_ids2': inputs2.get('token_type_ids', torch.zeros_like(inputs2['input_ids'])).squeeze(),
        }

# Collate function to combine items into batches.
def collate_func(batch):
    input_ids1 = torch.stack([item['input_ids1'] for item in batch])
    attention_mask1 = torch.stack([item['attention_mask1'] for item in batch])
    token_type_ids1 = torch.stack([item['token_type_ids1'] for item in batch])
    batch1 = {
        'input_ids': input_ids1,
        'attention_mask': attention_mask1,
        'token_type_ids': token_type_ids1
    }
    input_ids2 = torch.stack([item['input_ids2'] for item in batch])
    attention_mask2 = torch.stack([item['attention_mask2'] for item in batch])
    token_type_ids2 = torch.stack([item['token_type_ids2'] for item in batch])
    batch2 = {
        'input_ids': input_ids2,
        'attention_mask': attention_mask2,
        'token_type_ids': token_type_ids2
    }
    return batch1, batch2

# Custom BatchSampler that ensures each batch is drawn from one source only.
class RandomPerSourceBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        """
        dataset: instance of PerSourceContrastiveDataset.
        batch_size: desired batch size.
        drop_last: if True, drop the last batch of a source if it is smaller than batch_size.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        # Calculate source ranges using the dataset's cumulative lengths.
        self.source_ranges = []
        start = 0
        for cum_length in dataset.cum_lengths:
            self.source_ranges.append((start, cum_length))
            start = cum_length

    def __iter__(self):
        all_batches = []
        # For each data source, shuffle its indices and create batches.
        for (start, end) in self.source_ranges:
            indices = list(range(start, end))
            random.shuffle(indices)
            batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches = batches[:-1]
            all_batches.extend(batches)
        # Shuffle the order of batches (so sources are interleaved).
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for (start, end) in self.source_ranges:
            n = end - start
            if self.drop_last:
                total_batches += n // self.batch_size
            else:
                total_batches += (n + self.batch_size - 1) // self.batch_size
        return total_batches

# Function to create train and test DataLoaders using the above dataset and batch sampler.
def get_contrastive_dataloader(data_source, tokenizer, batch_size=32, max_length=512,
                                train_frac=0.99, drop_last=False, reshuffle_each_epoch=True, shuffle_seed=None):
    # Create train and test datasets.
    train_dataset = PerSourceContrastiveDataset(data_source, tokenizer, split='train',
                                                train_frac=train_frac, max_length=max_length,
                                                initial_shuffle=True, shuffle_seed=shuffle_seed)
    test_dataset = PerSourceContrastiveDataset(data_source, tokenizer, split='test',
                                               train_frac=train_frac, max_length=max_length,
                                               initial_shuffle=True, shuffle_seed=shuffle_seed)
    # Create custom batch samplers.
    train_batch_sampler = RandomPerSourceBatchSampler(train_dataset, batch_size, drop_last=drop_last)
    test_batch_sampler = RandomPerSourceBatchSampler(test_dataset, batch_size, drop_last=drop_last)

    # Optionally, if you want to reshuffle each epoch, you can call reshuffle() on the dataset manually.
    # For example, in your training loop at the start of each epoch:
    #     train_dataset.reshuffle(shuffle_seed=shuffle_seed)
    #     test_dataset.reshuffle(shuffle_seed=shuffle_seed)

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_func)
    test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, collate_fn=collate_func)

    return train_loader, test_loader

# Example test code
if __name__ == "__main__":
    # Create a sample DataFrame simulating a two-column CSV.
    data = {
        'sentence1': [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ],
        'sentence2': [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]
    }
    df = pd.DataFrame(data)
    
    # Simulate a second source by copying and modifying the first source.
    data = {
        'sentence1': [
            "a",
            "b",
            "c","d","e","f","g","h","i"
        ],
        'sentence2': [
            "a",
            "b",
            "c","d","e","f","g","h","i"
        ]
    }
    df2 = pd.DataFrame(data)
    
    # Combine the two sources by concatenation. The first half will be source 1 and the second half source 2.
    # combined_df = pd.concat([df, df2], axis=0).reset_index(drop=True)
    
    # Assume load_tokenizer() returns a valid tokenizer.
    from tokenizer_loader import load_tokenizer
    tokenizer = load_tokenizer()
    
    # Get the train and test DataLoaders.
    train_loader, test_loader = get_contrastive_dataloader(
        [df, df2],
        tokenizer,
        batch_size=3,
        train_frac=0.8,
        drop_last=False,
        max_length=5,
        reshuffle_each_epoch=True,  # Use this flag to reshuffle within your training loop.
        shuffle_seed=42
    )
    
    print("Train Loader Batches:")
    for batch1, batch2 in train_loader:
        print("Batch - Sentence1 IDs:", batch1['input_ids'][:,1])
        # print("Batch - Sentence2 IDs:", batch2['input_ids'])
        print("---")
    
    print("Test Loader Batches:")
    for batch1, batch2 in test_loader:
        print("Batch - Sentence1 IDs:", batch1['input_ids'][:,1])
        # print("Batch - Sentence2 IDs:", batch2['input_ids'])
        print("---")
