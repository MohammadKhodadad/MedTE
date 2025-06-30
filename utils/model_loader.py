from transformers import AutoModel, AutoModelForMaskedLM
import torch
import torch.nn as nn
import os

from peft import LoraConfig, get_peft_model

class Model(nn.Module):  # Inherit from torch.nn.Module
    def __init__(self, hf_address="bert-base-uncased", cache_dir='./cache',task='contrastive', peft_r=None,grad_checkpointing=False):
        super(Model, self).__init__()
        self.hf_address = hf_address
        self.cache_dir = cache_dir
        if task=='contrastive':
            self.model = AutoModel.from_pretrained(hf_address, cache_dir=cache_dir, trust_remote_code=True)
            if peft_r:
                lora_config = LoraConfig(
                    r=peft_r,              # low-rank dimension
                    lora_alpha=32,    # scaling
                    lora_dropout=0.1,
                    bias="none",
                    task_type="FEATURE_EXTRACTION"  # For a BERT-style encoder
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
        elif task=='mlm':
            self.model = AutoModelForMaskedLM.from_pretrained(hf_address, cache_dir=cache_dir, trust_remote_code=True)
            
        else:
            raise Exception('task has to be mlm or contastive.')
        if grad_checkpointing:
            self.model.gradient_checkpointing_enable()
    def encode(self, inputs_, use_cls=True):
        # Forward pass to get the model outputs
        outputs = self.model(**inputs_)

        # Extract the token embeddings from the model's output
        token_embeddings = outputs.last_hidden_state
        if use_cls:
            cls_embedding = token_embeddings[:, 0, :]
            return cls_embedding
        else:

            attention_mask = inputs_['attention_mask'].unsqueeze(-1)
            masked_embeddings = token_embeddings * attention_mask
            mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
            return mean_embedding
    # def _encode_chunk(self, inputs_, use_cls=True):
    #     """
    #     Encodes a small batch (chunk) of inputs.
    #     """
    #     outputs = self.model(**inputs_)  # Forward pass
    #     token_embeddings = outputs.last_hidden_state

    #     if use_cls:
    #         # Return CLS token (index 0) for each sequence
    #         cls_embedding = token_embeddings[:, 0, :]
    #         return cls_embedding
    #     else:
    #         attention_mask = inputs_['attention_mask'].unsqueeze(-1)  # [B, seq_len, 1]
    #         masked_embeddings = token_embeddings * attention_mask
    #         # Compute mean over valid tokens
    #         mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
    #         return mean_embedding

    # def encode(self, inputs_, use_cls=True, chunk_size=128):
    #     """
    #     Encodes inputs into embeddings. If the batch size exceeds `chunk_size`,
    #     it will split the data into chunks to avoid OOM issues.
    #     """
    #     # We'll assume inputs_ is a dictionary containing:
    #     #   - 'input_ids': [batch_size, seq_len]
    #     #   - 'attention_mask': [batch_size, seq_len]
    #     # (optionally 'token_type_ids' or other fields)

    #     batch_size = inputs_['input_ids'].shape[0]

    #     # If the batch size is small enough, just do one pass
    #     if batch_size <= chunk_size:
    #         return self._encode_chunk(inputs_, use_cls=use_cls)

    #     # Otherwise, split into chunks
    #     all_embeddings = []
    #     for start_idx in range(0, batch_size, chunk_size):
    #         end_idx = start_idx + chunk_size

    #         # Build a sub-dictionary for this chunk
    #         chunk_inputs = {
    #             'input_ids': inputs_['input_ids'][start_idx:end_idx],
    #             'attention_mask': inputs_['attention_mask'][start_idx:end_idx],
    #         }

    #         # If token_type_ids or other fields exist, slice them too
    #         if 'token_type_ids' in inputs_:
    #             chunk_inputs['token_type_ids'] = inputs_['token_type_ids'][start_idx:end_idx]

    #         # Encode the chunk
    #         chunk_embeddings = self._encode_chunk(chunk_inputs, use_cls=use_cls)
    #         all_embeddings.append(chunk_embeddings)

    #     # Concatenate all chunk embeddings along the batch dimension
    #     return torch.cat(all_embeddings, dim=0)

    def save_weights(self, save_path):
        if  not os.path.exists('./weights') :
            os.makedirs('./weights')
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved at {save_path}.")

    def load_weights(self, load_path):
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))
            print(f"Model weights loaded from {load_path}.")
        else:
            print(f"Weight file {load_path} does not exist.")
    def save_pretrained(self, address):
        self.model.save_pretrained(address)

    def forward(self, **inputs_):
        outputs = self.model(**inputs_)
        return outputs
if __name__ == "__main__":
    model_instance = Model()  # The model will be cached in ./cache
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_instance = model_instance.to(device)
    from tokenizer_loader import load_tokenizer
    tokenizer = load_tokenizer()
    inputs = tokenizer("Medical BERT is a powerful tool.", return_tensors="pt", padding=True, truncation=True)
    print(inputs.keys())
    inputs={key:inputs[key].to(device) for key in inputs.keys()}
    embedding = model_instance.encode(inputs)
    print(f"Encoded embedding: {embedding}")
    model_instance.save_weights("./weights/saved_model_weights.pth")
    model_instance.load_weights("./weights/saved_model_weights.pth")
    another_inputs = tokenizer("Another test sentence.", return_tensors="pt", padding=True, truncation=True)
    another_inputs={key:another_inputs[key].to(device) for key in another_inputs.keys()}
    another_embedding = model_instance.encode(another_inputs)
    print(f"Another encoded embedding: {another_embedding}")
