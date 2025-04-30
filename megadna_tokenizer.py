import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define the custom tokenizer for nucleotide sequences
nucleotides = ['**', 'A', 'T', 'C', 'G', 'N', '#']  # Define custom vocabulary for tokenization

class NucleotideTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {nt: idx for idx, nt in enumerate(vocab)}  # Map nucleotides to ids
        self.id_to_token = {idx: nt for idx, nt in enumerate(vocab)}  # Map ids to nucleotides

    def encode(self, sequence):
        # Handle unknown nucleotides by replacing them with 'N'
        return [self.token_to_id.get(nt, self.token_to_id['N']) for nt in sequence]  # Convert sequence to token ids

    def decode(self, token_ids):
        # Convert tensor values to Python integers if needed
        return ''.join([self.id_to_token[token_id.item() if hasattr(token_id, 'item') else token_id] for token_id in token_ids])  # Convert token ids back to sequence
    
    def save_pretrained(self, path):
        # Simple implementation to save the tokenizer
        import os
        import json
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump({"vocab": self.vocab}, f)

# Custom Dataset for training the model
class GenomicDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Tokenize the sequence and return as input_ids
        encoded = self.tokenizer.encode(sequence)
        encoded = encoded[:self.max_length] + [0] * (self.max_length - len(encoded))  # Padding/truncation
        return torch.tensor(encoded)

# Function to tokenize sequences
def tokenize_sequences(sequences):
    tokenized_sequences = []
    for seq in sequences:
        tokenized_sequences.append([nt for nt in seq])  # Tokenization by nucleotide
    return tokenized_sequences

# Function to generate sequences
def generate_sequence(model, tokenizer, seed_sequence, max_length=512, device='cuda'):
    model.eval()
    
    # Convert string seed to token IDs
    input_ids = tokenizer.encode(seed_sequence)  # Tokenize the seed sequence
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)  # Add batch dimension
    
    # Check if the model is MEGADNA class which has different generate parameters
    if hasattr(model, '__class__') and model.__class__.__name__ == 'MEGADNA':
        # Use the parameters that MEGADNA.generate() accepts - matching the example code
        generated_ids = model.generate(
            prime=input_ids,
            seq_len=max_length,
            temperature=0.95,
            filter_thres=0.0  # Using 0.0 as in the example code
        )
        
        # Convert token IDs to nucleotides directly
        generated_sequence = ''.join([tokenizer.id_to_token[token_id.item() if hasattr(token_id, 'item') else token_id]
                                     for token_id in generated_ids[0]])
    else:
        # For other models like GPT2LMHeadModel
        generated_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.95,
            top_k=50,
            top_p=0.95
        )
        generated_sequence = tokenizer.decode(generated_ids[0])
        
    return generated_sequence