from torch.utils.data import Dataset, DataLoader, Sampler
import random
import os, torch, pandas as pd
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor





class PerSourceContrastiveDataset(Dataset):
    """
    One CSV / DataFrame = one “source”.
    • EVERY row is embedded once with all-MiniLM-L6-v2 and cached.
    • __getitem__ ➜ tokenised tensors via `self.tokenizer`.
    • get_embedding_pair(idx) ➜ Tensor(shape=(2, 384)) from the cache.
    """
    def __init__(
        self,
        data_source      : Union[str, pd.DataFrame, List[pd.DataFrame]],
        tokenizer        : Optional[object] = None,  # e.g. AutoTokenizer
        cache_dir        : Optional[str]  = None,
        device           : str           = "cpu",
        overwrite_cache  : bool          = False,
        max_length       : int           = 512,
        initial_shuffle  : bool          = True,
        shuffle_seed     : Optional[int] = None,
        encode_batch_size: int           = 256,
    ):
        self.device            = device
        self.max_length        = max_length
        self.cache_dir         = cache_dir or (data_source if isinstance(data_source, str) else ".")
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1) Load the sentence-transformers model (includes its own tokenizer+pooling)
        self._st_model         = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.encode_batch_size = encode_batch_size

        # 2) Use external tokenizer if provided, else fall back to the ST model’s tokenizer
        self.tokenizer         = tokenizer or self._st_model.tokenizer

        # 3) Load & optionally shuffle each CSV / DataFrame
        self.sources = self._gather_sources(data_source, initial_shuffle, shuffle_seed)

        # 4) Embed all rows once and cache
        self.dataframes, self.embeddings = [], []
        for sidx, df in enumerate(self.sources):
            df_cached, emb = self._maybe_cache_all(df, sidx, overwrite_cache)
            self.dataframes.append(df_cached)
            self.embeddings.append(emb)  # Tensor(N, 2, 384)

        # 5) Cumulative lengths for fast index lookup
        self.cum = torch.tensor([len(df) for df in self.dataframes]).cumsum(0)

    def _gather_sources(self, ds, shuffle: bool, seed: Optional[int]):
        def _prep(df: pd.DataFrame):
            df = df.dropna().drop_duplicates("sentence1").reset_index(drop=True)
            if shuffle:
                df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            return df

        if isinstance(ds, str):
            csvs = sorted(f for f in os.listdir(ds) if f.endswith(".csv"))
            frames = []
            for f in csvs:
                path = os.path.join(ds, f)
                try:
                    frames.append(_prep(pd.read_csv(path)))
                except Exception as e:
                    print(f"Failed to load {f}: {e}")
            return frames

        if isinstance(ds, pd.DataFrame):
            return [_prep(ds)]
        if isinstance(ds, list):
            return [_prep(d) for d in ds]
        raise ValueError("Unsupported data_source type")

    def _encode_batch(self, texts: List[str]) -> Tensor:
        # Use the SentenceTransformer wrapper for correct pooling & normalization
        return self._st_model.encode(
            texts,
            batch_size=self.encode_batch_size,
            convert_to_tensor=True,
            show_progress_bar=False
        ).cpu()

    def _maybe_cache_all(self, df: pd.DataFrame, idx: int, overwrite: bool):
        # cache_path = os.path.join(self.cache_dir, f"src{idx}.emb.pt")
        # if os.path.exists(cache_path) and not overwrite:
        #     emb = torch.load(cache_path, map_location="cpu")
        # else:
        emb1, emb2 = [], []
        for i in tqdm(range(0, len(df), self.encode_batch_size), desc=f"src{idx}"):
            batch = df.iloc[i : i + self.encode_batch_size]
            emb1.append(self._encode_batch(batch["sentence1"].tolist()))
            emb2.append(self._encode_batch(batch["sentence2"].tolist()))
        emb = torch.stack([torch.cat(emb1), torch.cat(emb2)], dim=1)  # (N, 2, 384)
            # torch.save(emb, cache_path)  # uncomment to persist cache
        return df, emb

    def __len__(self):
        return int(self.cum[-1])

    def _loc(self, idx: int):
        # ensure idx is a Python int
        if torch.is_tensor(idx):
            idx = idx.item()
        elif hasattr(idx, "__int__"):
            idx = int(idx)

        src = torch.searchsorted(self.cum, torch.tensor(idx), right=True).item()
        offset = 0 if src == 0 else self.cum[src - 1].item()
        row = idx - offset
        return src, row

    def __getitem__(self, idx: int):
        src, row = self._loc(idx)
        row_series = self.dataframes[src].iloc[row]
        s1, s2     = row_series["sentence1"], row_series["sentence2"]

        # Tokenize both sentences with the chosen tokenizer
        tok1 = self.tokenizer(
            s1, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        tok2 = self.tokenizer(
            s2, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids1"      : tok1["input_ids"].squeeze(),
            "attention_mask1" : tok1["attention_mask"].squeeze(),
            "token_type_ids1" : tok1.get("token_type_ids", torch.zeros_like(tok1["input_ids"])).squeeze(),
            "input_ids2"      : tok2["input_ids"].squeeze(),
            "attention_mask2" : tok2["attention_mask"].squeeze(),
            "token_type_ids2" : tok2.get("token_type_ids", torch.zeros_like(tok2["input_ids"])).squeeze(),
        }

    def get_embedding_pair(self, idx: int) -> Tensor:
        """Return the cached (emb1, emb2) tensor pair — shape (2, 384)."""
        src, row = self._loc(idx)
        return self.embeddings[src][row]


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

import torch, random
from torch.utils.data import Sampler
from sklearn.cluster import MiniBatchKMeans   # scikit‑learn ≥0.24
import numpy as np
def to_py_int(x):
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError("Can only convert 1-element tensors to int")
        return x.cpu().item()
    return int(x)

class LocalityPerSourceBatchSampler(Sampler):
    """
    • Each epoch draws a fresh random Gaussian vector g ∈ R^D.
    • For every source, rows are sorted by score = ⟨embedding, g⟩,
      then cut into consecutive chunks of `batch_size`.
    • Batches themselves are shuffled so that sources interleave.
    """
    def __init__(self, dataset, batch_size: int, super_size: int = 16000, drop_last: bool = False):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.super_size  = super_size
        self.counter = -1
        self.drop_last   = drop_last
        if super_size < batch_size:
            raise ValueError("super_size must be ≥ batch_size")
        # source_ranges: (global_start, global_end) for every source
        self.source_ranges = []
        start = 0
        for cum in dataset.cum:
            self.source_ranges.append((start, cum))
            start = cum

        # assume all embeddings share same dimensionality
        self.emb_dim = dataset.embeddings[0].shape[-1]

    # -------------------------------------------------------------
    def __iter__(self):
        """
        Epoch‑wise locality batching via MiniBatchKMeans.

        • For each source  s :
            1.  Pick  K_s  = ceil(N_s / batch_size) clusters.
            2.  Run MiniBatchKMeans on its embeddings (fast, online).
            3.  Gather row indices by cluster id.
            4.  Inside each cluster, shuffle row order → slice into
                fixed‑width mini‑batches.
        • Finally, shuffle the list of all batches so that sources interleave.
        """

        
        self.counter += 2
        
        all_batches = []

        for src_idx, (start, end) in enumerate(self.source_ranges):
            n_src = end - start
            if n_src <= self.batch_size:           # tiny source → one batch
                all_batches.append(list(range(start, end)))
                continue

            # ---------- 1.  run K‑Means on this source ---------------------------
            # emb = self.dataset.embeddings[src_idx][:, 0].numpy()   # (N_src, D)
            emb = self.dataset.embeddings[src_idx][:, 1].cpu().numpy()
            max_k = max(1, n_src // self.super_size)

            # grow k by 1 each epoch, but cap at max_k
            k = min(self.counter, max_k)
            kmeans = MiniBatchKMeans(n_clusters=to_py_int(k),
                                    batch_size=8192,
                                    n_init=1,
                                    max_iter=20,
                                    random_state=None)
            cluster_id = kmeans.fit_predict(emb)                   # (N_src,)

            # ---------- 2.  bucket indices by cluster ----------------------------
            buckets = {}
            for local_idx, cid in enumerate(cluster_id):
                buckets.setdefault(cid, []).append(local_idx + start)  # global idx

            # ---------- 3.  slice each cluster into batches ----------------------
            for idxs in buckets.values():
                random.shuffle(idxs)                   # intra‑cluster shuffle
                for i in range(0, len(idxs), self.batch_size):
                    chunk = idxs[i:i + self.batch_size]
                    if len(chunk) == self.batch_size:
                        all_batches.append(chunk)
                    elif not self.drop_last:           # keep a short tail batch?
                        all_batches.append(chunk)

        random.shuffle(all_batches)                    # interleave sources
        for batch in all_batches:
            yield batch


    # -------------------------------------------------------------
    def __len__(self):
        tot = 0
        for (s, e) in self.source_ranges:
            n = e - s
            if self.drop_last:
                tot += n // self.batch_size
            else:
                tot += (n + self.batch_size - 1) // self.batch_size
        return tot



# Function to create train and test DataLoaders using the above dataset and batch sampler.
from torch.utils.data import DataLoader, SubsetRandomSampler

from torch.utils.data import DataLoader

def get_contrastive_dataloader(
        data_source,
        tokenizer,
        batch_size      = 32,
        device          = 'cuda',
        max_length      = 512,
        drop_last       = False,
        shuffle_seed    = None):

    # a) full dataset (embeds + caches all rows once)
    dataset  = PerSourceContrastiveDataset(
        data_source       = data_source,
        tokenizer         = tokenizer,
        device            = device,
        max_length        = max_length,
        initial_shuffle   = True,
        shuffle_seed      = shuffle_seed)

    # b) locality‑aware batch sampler (NO allowed_idx)
    sampler  = LocalityPerSourceBatchSampler(
        dataset, batch_size=batch_size, drop_last=drop_last)

    # c) DataLoader
    loader   = DataLoader(dataset,
                          batch_sampler = sampler,
                          collate_fn    = collate_func,
                          num_workers   = 0)

    return loader



# Example test code
if __name__ == "__main__":
    
    # ds = PerSourceContrastiveDataset("toy_data/", device="cpu")
    # loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    # for batch in loader:          # your training loop
    #     ...

    # # Need a pre‑computed vector later?
    # vecs = ds.get_embedding_pair(42)   # returns tensor(2, 768)
    from tokenizer_loader import load_tokenizer
    tokenizer = load_tokenizer()
    train_loader = get_contrastive_dataloader(
        data_source="toy_data/",
        device = 'cuda',
        tokenizer=tokenizer,
        batch_size=8)

    print("Train Loader Batches:")
    for batch1, batch2 in train_loader:
        print("Batch - Sentence1 IDs:", batch1['input_ids'][:,1])
        # print("Batch - Sentence2 IDs:", batch2['input_ids'])
        print("---")
    
