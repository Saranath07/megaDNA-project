import torch
from megadna_tokenizer import NucleotideTokenizer, nucleotides

def generate_dna_sequence(
    model_path,
    seed_sequence,
    seq_length=200,
    temperature=0.8,
    filter_threshold=0.9
):
    # Determine device
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

if __name__ == "__main__":
    # Generate a short sequence and print it
    seed = "ATGCGTACGTAGC"
    generated_seq = generate_dna_sequence(
        model_path="./fine_tuned_megaDNA_model.pt",
        seed_sequence=seed,
        seq_length=200,
        temperature=0.9
    )
    
    print("\nGenerated DNA sequence:")
    print(">generated_dna_sequence")
    print(generated_seq)
    
    # Save to file
    output_file = "generated_sequence_output.fasta"
    with open(output_file, 'w') as f:
        f.write(f">generated_dna_sequence_seed_{seed}_temp_0.9\n")
        f.write(generated_seq)
    
    print(f"\nSequence also saved to {output_file}")