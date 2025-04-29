import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from megadna_tokenizer import NucleotideTokenizer, nucleotides

def generate_dna_sequence(
    model_path,
    seed_sequence,
    seq_length=500,
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

def analyze_sequence(sequence):
    """Analyze the nucleotide composition of a DNA sequence"""
    # Count nucleotides
    nucleotide_counts = Counter(sequence)
    total_count = len(sequence)
    
    # Calculate percentages
    percentages = {nt: (count / total_count) * 100 for nt, count in nucleotide_counts.items()}
    
    # Calculate GC content
    gc_content = ((nucleotide_counts.get('G', 0) + nucleotide_counts.get('C', 0)) / total_count) * 100
    
    return {
        'counts': nucleotide_counts,
        'percentages': percentages,
        'gc_content': gc_content,
        'length': total_count
    }

def plot_nucleotide_distribution(analysis_results, title="Nucleotide Distribution"):
    """Plot the nucleotide distribution as a bar chart"""
    nucleotides = ['A', 'T', 'G', 'C']
    counts = [analysis_results['counts'].get(nt, 0) for nt in nucleotides]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(nucleotides, counts)
    
    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        percentage = analysis_results['percentages'].get(nucleotides[i], 0)
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{percentage:.1f}%', ha='center')
    
    plt.title(title)
    plt.xlabel('Nucleotide')
    plt.ylabel('Count')
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    print(f"Plot saved as {title.replace(' ', '_').lower()}.png")

def find_repeats(sequence, min_length=5):
    """Find repeating patterns in the sequence"""
    repeats = {}
    seq_length = len(sequence)
    
    for pattern_length in range(min_length, 21):  # Check patterns up to 20 nucleotides
        for i in range(seq_length - pattern_length + 1):
            pattern = sequence[i:i+pattern_length]
            if pattern in repeats:
                repeats[pattern] += 1
            else:
                # Count occurrences of this pattern in the whole sequence
                occurrences = sequence.count(pattern)
                if occurrences > 1:  # Only store if it appears more than once
                    repeats[pattern] = occurrences
    
    # Sort by number of occurrences
    sorted_repeats = sorted(repeats.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_repeats[:10]  # Return top 10 repeats

def compare_sequences(sequences, names):
    """Compare multiple sequences and visualize their nucleotide composition"""
    analyses = [analyze_sequence(seq) for seq in sequences]
    
    # Plot nucleotide percentages for each sequence
    nucleotides = ['A', 'T', 'G', 'C']
    percentages = [[analysis['percentages'].get(nt, 0) for nt in nucleotides] for analysis in analyses]
    
    x = np.arange(len(nucleotides))
    width = 0.2
    
    plt.figure(figsize=(12, 8))
    
    for i, (pct, name) in enumerate(zip(percentages, names)):
        offset = width * (i - len(percentages)/2 + 0.5)
        bars = plt.bar(x + offset, pct, width, label=name)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Nucleotide')
    plt.ylabel('Percentage')
    plt.title('Nucleotide Composition Comparison')
    plt.xticks(x, nucleotides)
    plt.legend()
    plt.savefig("sequence_comparison.png")
    print("Comparison plot saved as sequence_comparison.png")

def save_to_fasta(sequences, descriptions, output_file):
    """Save sequences to a FASTA file"""
    records = []
    for i, (seq, desc) in enumerate(zip(sequences, descriptions)):
        record = SeqRecord(
            Seq(seq),
            id=f"seq_{i+1}",
            description=desc
        )
        records.append(record)
    
    with open(output_file, "w") as handle:
        SeqIO.write(records, handle, "fasta")
    
    print(f"Sequences saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate and analyze DNA sequences from fine-tuned model')
    parser.add_argument('--model', type=str, default='./fine_tuned_megaDNA_model.pt', 
                        help='Path to the fine-tuned model')
    parser.add_argument('--seeds', type=str, nargs='+', default=['ATGC', 'GCTA', 'TAGC'], 
                        help='Seed sequences to use (can provide multiple)')
    parser.add_argument('--length', type=int, default=500, 
                        help='Length of sequences to generate')
    parser.add_argument('--temp', type=float, default=0.8, 
                        help='Temperature for generation')
    parser.add_argument('--output', type=str, default='analyzed_sequences.fasta', 
                        help='Output FASTA file')
    parser.add_argument('--analyze', action='store_true', 
                        help='Perform analysis on generated sequences')
    
    args = parser.parse_args()
    
    print(f"Using model: {args.model}")
    print(f"Generating sequences with seeds: {args.seeds}")
    
    # Generate sequences for each seed
    sequences = []
    descriptions = []
    
    for seed in args.seeds:
        print(f"\nGenerating sequence with seed {seed}...")
        seq = generate_dna_sequence(
            model_path=args.model,
            seed_sequence=seed,
            seq_length=args.length,
            temperature=args.temp
        )
        sequences.append(seq)
        descriptions.append(f"Generated with seed={seed}, temp={args.temp}, len={args.length}")
        
        # Print preview
        print(f"Sequence preview (first 50 bases): {seq[:50]}...")
        
        if args.analyze:
            print("\nAnalyzing sequence...")
            analysis = analyze_sequence(seq)
            print(f"Length: {analysis['length']} bp")
            print(f"Nucleotide counts: {analysis['counts']}")
            print(f"GC content: {analysis['gc_content']:.2f}%")
            
            # Find repeating patterns
            print("\nTop repeating patterns:")
            repeats = find_repeats(seq)
            for pattern, count in repeats:
                print(f"  {pattern}: {count} occurrences")
            
            # Plot nucleotide distribution for this sequence
            plot_nucleotide_distribution(analysis, f"Nucleotide Distribution - Seed {seed}")
    
    # Save all sequences to FASTA file
    save_to_fasta(sequences, descriptions, args.output)
    
    if args.analyze and len(sequences) > 1:
        # Compare all sequences
        print("\nComparing sequences...")
        compare_sequences(sequences, [f"Seed {seed}" for seed in args.seeds])

if __name__ == "__main__":
    main()