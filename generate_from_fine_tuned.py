import torch
import argparse
from megadna_tokenizer import NucleotideTokenizer, nucleotides

def generate_dna_sequence(
    model_path,
    seed_sequence,
    seq_length=1000,
    temperature=0.8,
    filter_threshold=0.9,
    device=None,
    num_sequences=1
):
    """
    Generate DNA sequences using a fine-tuned megaDNA model.
    
    Args:
        model_path: Path to the saved model
        seed_sequence: Seed DNA sequence to start generation
        seq_length: Total length of sequence to generate
        temperature: Controls randomness (lower = more deterministic)
        filter_threshold: Threshold for filtering logits
        device: Device to run on ('cuda' or 'cpu')
        num_sequences: Number of sequences to generate
    
    Returns:
        List of generated DNA sequences
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading fine-tuned model from {model_path}...")
    model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = NucleotideTokenizer(nucleotides)
    
    # Encode the seed sequence
    print(f"Seed sequence: {seed_sequence}")
    encoded_seed = tokenizer.encode(seed_sequence)
    seed_tensor = torch.tensor([encoded_seed], device=device)
    
    generated_sequences = []
    
    # Generate multiple sequences if requested
    for i in range(num_sequences):
        print(f"\nGenerating sequence {i+1}/{num_sequences} of length {seq_length} with temperature {temperature}...")
        with torch.no_grad():
            generated_tensor = model.generate(
                prime=seed_tensor,
                seq_len=seq_length,
                filter_thres=filter_threshold,
                temperature=temperature
            )
        
        # Decode the generated sequence
        generated_seq = tokenizer.decode(generated_tensor[0].tolist())
        generated_sequences.append(generated_seq)
        
        # Print a preview
        print(f"Sequence {i+1} preview (first 50 bases):\n{generated_seq[:50]}...")
    
    return generated_sequences

def save_sequences_to_fasta(sequences, seed, temperature, output_prefix="generated"):
    """Save generated sequences to a FASTA file"""
    output_file = f"{output_prefix}_sequences.fasta"
    
    with open(output_file, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">generated_dna_sequence_{i+1}_seed_{seed}_temp_{temperature}\n")
            # Write sequence with line breaks every 80 characters for FASTA format
            for j in range(0, len(seq), 80):
                f.write(f"{seq[j:j+80]}\n")
    
    print(f"\nAll sequences saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate DNA sequences using a fine-tuned megaDNA model')
    parser.add_argument('--model', type=str, default='./fine_tuned_megaDNA_model.pt', 
                        help='Path to the fine-tuned model (default: ./fine_tuned_megaDNA_model.pt)')
    parser.add_argument('--seed', type=str, default='ATGC', 
                        help='Seed DNA sequence (default: ATGC)')
    parser.add_argument('--length', type=int, default=1000, 
                        help='Length of sequence to generate (default: 1000)')
    parser.add_argument('--temp', type=float, default=0.8, 
                        help='Temperature for generation (0.0-1.0, lower = more deterministic) (default: 0.8)')
    parser.add_argument('--num', type=int, default=1, 
                        help='Number of sequences to generate (default: 1)')
    parser.add_argument('--output', type=str, default='generated', 
                        help='Output file prefix (default: "generated")')
    
    args = parser.parse_args()
    
    # Generate the sequences
    generated_sequences = generate_dna_sequence(
        model_path=args.model,
        seed_sequence=args.seed,
        seq_length=args.length,
        temperature=args.temp,
        num_sequences=args.num
    )
    
    # Save to FASTA file
    output_file = save_sequences_to_fasta(
        generated_sequences, 
        args.seed, 
        args.temp, 
        args.output
    )
    
    print(f"\nGeneration complete! {len(generated_sequences)} sequences generated.")
    print(f"To view the full sequences, check the file: {output_file}")

if __name__ == "__main__":
    main()