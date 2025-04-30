import torch
import os
import numpy as np
from torch.serialization import add_safe_globals
from megaDNA.megadna import MEGADNA
from megadna_tokenizer import NucleotideTokenizer, nucleotides

def save_sequence_to_fasta(sequence, file_path, header="Generated DNA sequence"):
    """
    Save a DNA sequence to a FASTA format file
    
    Args:
        sequence (str): The DNA sequence to save
        file_path (str): Path to save the FASTA file
        header (str): Header for the FASTA sequence
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(f">{header}\n")
        # Write sequence in lines of 80 characters (standard FASTA format)
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")
    print(f"Sequence saved to {file_path}")

def token2nucleotide(s):
    """Convert token ID to nucleotide"""
    return nucleotides[s]

def generate_dna_sequences(num_sequences=1000, seq_length=1000):
    """
    Generate DNA sequences using both pretrained and fine-tuned models
    
    Args:
        num_sequences (int): Number of sequences to generate
        seq_length (int): Length of each sequence
    """
    # Setup output directory
    output_dir = "generated_sequences"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = NucleotideTokenizer(nucleotides)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Add MEGADNA to safe globals list for unpickling
    add_safe_globals([MEGADNA])
    
    # Define model paths
    pretrained_model_path = "notebook/megaDNA_phage_145M.pt"
    finetuned_model_path = "progressive_fine_tuned_megaDNA_model.pt"
    
    # Generate sequences with pretrained model
    print("\n===== GENERATING SEQUENCES WITH PRETRAINED MODEL =====")
    generate_with_model(pretrained_model_path, "pretrained", num_sequences, seq_length, device, output_dir)
    
    # Generate sequences with fine-tuned model
    print("\n===== GENERATING SEQUENCES WITH FINE-TUNED MODEL =====")
    generate_with_model(finetuned_model_path, "finetuned", num_sequences, seq_length, device, output_dir)

def generate_with_model(model_path, model_type, num_sequences, seq_length, device, output_dir):
    """
    Generate sequences with a specific model
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model (pretrained or finetuned)
        num_sequences (int): Number of sequences to generate
        seq_length (int): Length of each sequence
        device (str): Device to use (cuda or cpu)
        output_dir (str): Directory to save generated sequences
    """
    try:
        # Load the model
        print(f"Loading {model_type} model from {model_path}...")
        model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        model.eval()
        
        for i in range(num_sequences):
            # Generate a random primer sequence
            primer_length = 4  # Short primer length
            primer_sequence = torch.tensor(np.random.choice(np.arange(1, 5), primer_length)).long().to(device)[None,]
            primer_dna = ''.join(map(token2nucleotide, primer_sequence[0]))
            
            print(f"Generating sequence {i+1}/{num_sequences} with primer: {primer_dna}")
            
            # Generate sequence
            seq_tokenized = model.generate(
                primer_sequence,
                seq_len=seq_length,
                temperature=0.95,
                filter_thres=0.0
            )
            
            # Convert tokens to nucleotides
            generated_sequence = ''.join(map(token2nucleotide, seq_tokenized.squeeze().cpu().int()))
            
            # Split by '#' character (end token) and take the first part
            clean_sequence = generated_sequence.split('#')[0]
            
            # Save to FASTA file
            file_path = os.path.join(output_dir, f"{model_type}_sequence_{i+1}.fasta")
            save_sequence_to_fasta(
                clean_sequence,
                file_path,
                header=f"{model_type.capitalize()} model generated sequence {i+1} (primer: {primer_dna})"
            )
            
            # Print statistics
            if i == 0:  # Only for the first sequence
                nucleotide_counts = {
                    'A': clean_sequence.count('A'),
                    'T': clean_sequence.count('T'),
                    'G': clean_sequence.count('G'),
                    'C': clean_sequence.count('C'),
                    'Other': len(clean_sequence) - (clean_sequence.count('A') + clean_sequence.count('T') + 
                                                clean_sequence.count('G') + clean_sequence.count('C'))
                }
                print(f"{model_type.capitalize()} model nucleotide distribution:")
                for nuc, count in nucleotide_counts.items():
                    percentage = (count / len(clean_sequence)) * 100 if len(clean_sequence) > 0 else 0
                    print(f"{nuc}: {count} ({percentage:.2f}%)")
        
        # Clean up to free memory
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error generating sequences with {model_type} model: {str(e)}")

if __name__ == "__main__":
    # Generate 5 sequences of 1000 base pairs each
    # Change num_sequences if you want more sequences
    generate_dna_sequences(num_sequences=5, seq_length=10000)