from transformers import AutoTokenizer

def load_tokenizer(hf_address="bert-base-uncased",cache_dir='./cache'):
    return AutoTokenizer.from_pretrained("bert-base-uncased",cache_dir=cache_dir)

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    text = "This is a test sentence for the tokenizer."
    tokens = tokenizer.tokenize(text)

    print(f"Original text: {text}")
    print(f"Tokenized text: {tokens}")
    
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded_text = tokenizer.decode(token_ids)

    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: {decoded_text}")