import torch
import argparse
from megadna_tokenizer import NucleotideTokenizer, nucleotides

def generate_dna_sequence(
    model_path,
    seed_sequence,
    seq_length=1000,
    temperature=0.8,
    filter_threshold=0.9,
    device=None
):
    """
    Generate a DNA sequence using a trained megaDNA model.
    
    Args:
        model_path: Path to the saved model
        seed_sequence: Seed DNA sequence to start generation
        seq_length: Total length of sequence to generate
        temperature: Controls randomness (lower = more deterministic)
        filter_threshold: Threshold for filtering logits
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Generated DNA sequence as a string
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = NucleotideTokenizer(nucleotides)
    
    # Encode the seed sequence
    print(f"Seed sequence: {seed_sequence}")
    encoded_seed = tokenizer.encode(seed_sequence)
    seed_tensor = torch.tensor([encoded_seed], device=device)
    
    # Generate sequence
    print(f"Generating sequence of length {seq_length} with temperature {temperature}...")
    with torch.no_grad():
        generated_tensor = model.generate(
            prime=seed_tensor,
            seq_len=seq_length,
            filter_thres=filter_threshold,
            temperature=temperature
        )
    
    # Decode the generated sequence
    generated_seq = tokenizer.decode(generated_tensor[0].tolist())
    
    return generated_seq

def main():
    parser = argparse.ArgumentParser(description='Generate DNA sequences using a trained megaDNA model')
    parser.add_argument('--model', type=str, default='./fine_tuned_megaDNA_model.pt', help='Path to the model')
    parser.add_argument('--seed', type=str, default='ATGC', help='Seed DNA sequence')
    parser.add_argument('--length', type=int, default=1000, help='Length of sequence to generate')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature for generation')
    parser.add_argument('--output', type=str, default='generated_sequence.fasta', help='Output file path')
    
    args = parser.parse_args()
    
    # Generate the sequence
    generated_seq = generate_dna_sequence(
        model_path=args.model,
        seed_sequence=args.seed,
        seq_length=args.length,
        temperature=args.temp
    )
    
    # Print a preview
    print(f"\nGenerated sequence preview (first 100 bases):\n{generated_seq[:100]}...")
    
    # Save to FASTA file
    with open(args.output, 'w') as f:
        f.write(f">generated_dna_sequence_seed_{args.seed}_temp_{args.temp}\n")
        f.write(generated_seq)
    
    print(f"\nFull sequence saved to {args.output}")

if __name__ == "__main__":
    main()