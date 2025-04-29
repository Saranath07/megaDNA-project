import torch
import argparse
import time
from megadna_tokenizer import NucleotideTokenizer, nucleotides
from Bio.SeqUtils import gc_fraction
from Bio.Seq import Seq

def generate_dna_sequence(
    model_path,
    seed_sequence,
    seq_length=300,
    temperature=0.8,
    filter_threshold=0.9,
    device=None
):
    """Generate a DNA sequence using the fine-tuned model"""
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the model
    model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = NucleotideTokenizer(nucleotides)
    
    # Encode the seed sequence
    encoded_seed = tokenizer.encode(seed_sequence)
    seed_tensor = torch.tensor([encoded_seed], device=device)
    
    # Generate sequence
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

def calculate_gc_content(sequence):
    """Calculate the GC content of a sequence"""
    return gc_fraction(sequence) * 100

def has_restriction_site(sequence, site):
    """Check if the sequence contains a specific restriction site"""
    return site in sequence

def has_start_codon(sequence):
    """Check if the sequence contains a start codon (ATG)"""
    return "ATG" in sequence

def has_stop_codon(sequence):
    """Check if the sequence contains a stop codon (TAA, TAG, TGA)"""
    stop_codons = ["TAA", "TAG", "TGA"]
    return any(codon in sequence for codon in stop_codons)

def has_open_reading_frame(sequence, min_length=50):
    """Check if the sequence contains an open reading frame of at least min_length"""
    seq_obj = Seq(sequence)
    
    # Check all three reading frames
    for i in range(3):
        # Get the sequence in this reading frame
        frame = seq_obj[i:]
        
        # Find start codons
        for j in range(0, len(frame), 3):
            if j+3 > len(frame):
                break
                
            # If we find a start codon
            if frame[j:j+3] == "ATG":
                # Look for a stop codon
                for k in range(j+3, len(frame), 3):
                    if k+3 > len(frame):
                        break
                        
                    if frame[k:k+3] in ["TAA", "TAG", "TGA"]:
                        # If we found a stop codon, check if the ORF is long enough
                        orf_length = k - j + 3
                        if orf_length >= min_length:
                            return True
    
    return False

def generate_sequences_with_criteria(
    model_path,
    seed_sequences,
    num_sequences=5,
    seq_length=300,
    temperature=0.8,
    gc_min=30,
    gc_max=70,
    require_restriction_sites=None,
    require_orf=False,
    max_attempts=100
):
    """Generate sequences that meet specific criteria"""
    if require_restriction_sites is None:
        require_restriction_sites = []
    
    successful_sequences = []
    attempts_per_seed = {}
    
    print(f"Generating {num_sequences} sequences with the following criteria:")
    print(f"- GC content between {gc_min}% and {gc_max}%")
    if require_restriction_sites:
        print(f"- Must contain restriction sites: {', '.join(require_restriction_sites)}")
    if require_orf:
        print(f"- Must contain an open reading frame")
    
    # Try each seed sequence
    for seed in seed_sequences:
        attempts_per_seed[seed] = 0
        sequences_from_this_seed = 0
        
        print(f"\nUsing seed sequence: {seed}")
        
        # Keep trying until we get enough sequences or reach max attempts
        while (sequences_from_this_seed < num_sequences // len(seed_sequences) + 1 and 
               attempts_per_seed[seed] < max_attempts):
            
            attempts_per_seed[seed] += 1
            
            # Generate a sequence
            sequence = generate_dna_sequence(
                model_path=model_path,
                seed_sequence=seed,
                seq_length=seq_length,
                temperature=temperature
            )
            
            # Check if it meets our criteria
            gc_content = calculate_gc_content(sequence)
            has_sites = all(has_restriction_site(sequence, site) for site in require_restriction_sites)
            has_orf = not require_orf or has_open_reading_frame(sequence)
            
            # If it meets all criteria, add it to our list
            if (gc_min <= gc_content <= gc_max and has_sites and has_orf):
                successful_sequences.append({
                    'sequence': sequence,
                    'seed': seed,
                    'gc_content': gc_content,
                    'has_start_codon': has_start_codon(sequence),
                    'has_stop_codon': has_stop_codon(sequence),
                    'has_orf': has_open_reading_frame(sequence)
                })
                sequences_from_this_seed += 1
                
                print(f"  Found sequence {len(successful_sequences)}/{num_sequences} "
                      f"(attempt {attempts_per_seed[seed]}):")
                print(f"  - GC content: {gc_content:.2f}%")
                print(f"  - Has start codon: {has_start_codon(sequence)}")
                print(f"  - Has stop codon: {has_stop_codon(sequence)}")
                print(f"  - Has ORF: {has_open_reading_frame(sequence)}")
                print(f"  - Preview: {sequence[:50]}...")
            
            # Stop if we have enough sequences
            if len(successful_sequences) >= num_sequences:
                break
        
        print(f"  Made {attempts_per_seed[seed]} attempts with seed {seed}, "
              f"found {sequences_from_this_seed} valid sequences")
    
    print(f"\nGenerated {len(successful_sequences)}/{num_sequences} sequences "
          f"meeting all criteria in {sum(attempts_per_seed.values())} total attempts")
    
    return successful_sequences

def save_to_fasta(sequences, output_file):
    """Save sequences to a FASTA file"""
    with open(output_file, 'w') as f:
        for i, seq_data in enumerate(sequences):
            sequence = seq_data['sequence']
            seed = seq_data['seed']
            gc = seq_data['gc_content']
            
            # Write header
            f.write(f">sequence_{i+1}_seed_{seed}_gc_{gc:.2f}\n")
            
            # Write sequence with line breaks every 80 characters
            for j in range(0, len(sequence), 80):
                f.write(f"{sequence[j:j+80]}\n")
    
    print(f"\nSequences saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate DNA sequences with specific properties')
    parser.add_argument('--model', type=str, default='./fine_tuned_megaDNA_model.pt', 
                        help='Path to the fine-tuned model')
    parser.add_argument('--seeds', type=str, nargs='+', default=['ATGC', 'GCTA', 'TAGC'], 
                        help='Seed sequences to use')
    parser.add_argument('--num', type=int, default=5, 
                        help='Number of sequences to generate')
    parser.add_argument('--length', type=int, default=300, 
                        help='Length of sequences to generate')
    parser.add_argument('--temp', type=float, default=0.9, 
                        help='Temperature for generation (higher = more random)')
    parser.add_argument('--gc-min', type=float, default=30, 
                        help='Minimum GC content percentage')
    parser.add_argument('--gc-max', type=float, default=70, 
                        help='Maximum GC content percentage')
    parser.add_argument('--restriction-sites', type=str, nargs='*', default=[], 
                        help='Required restriction sites')
    parser.add_argument('--require-orf', action='store_true', 
                        help='Require sequences to contain an open reading frame')
    parser.add_argument('--output', type=str, default='filtered_sequences.fasta', 
                        help='Output FASTA file')
    parser.add_argument('--max-attempts', type=int, default=100, 
                        help='Maximum number of generation attempts per seed')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Generate sequences meeting the criteria
    sequences = generate_sequences_with_criteria(
        model_path=args.model,
        seed_sequences=args.seeds,
        num_sequences=args.num,
        seq_length=args.length,
        temperature=args.temp,
        gc_min=args.gc_min,
        gc_max=args.gc_max,
        require_restriction_sites=args.restriction_sites,
        require_orf=args.require_orf,
        max_attempts=args.max_attempts
    )
    
    # Save to FASTA file
    if sequences:
        save_to_fasta(sequences, args.output)
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()