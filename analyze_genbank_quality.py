#!/usr/bin/env python3
"""
Script to analyze GenBank database file for sequence quality and GC content.
This script checks if the sequences in the GenBank file have appropriate GC content
and are proper DNA sequences without anomalies.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from Bio import SeqIO
from Bio import Seq
# Calculate GC content manually instead of using Bio.SeqUtils.GC
import os

def analyze_sequence(sequence):
    """Analyze the nucleotide composition of a DNA sequence"""
    # Convert sequence to string if it's a Seq object
    if not isinstance(sequence, str):
        sequence = str(sequence)
    
    # Count nucleotides
    nucleotide_counts = Counter(sequence.upper())
    total_count = len(sequence)
    
    # Calculate percentages
    percentages = {nt: (count / total_count) * 100 for nt, count in nucleotide_counts.items()}
    
    # Calculate GC content
    gc_content = ((nucleotide_counts.get('G', 0) + nucleotide_counts.get('C', 0)) / total_count) * 100
    
    # Check for non-standard nucleotides
    standard_nucleotides = set(['A', 'T', 'G', 'C', 'N'])
    non_standard = [nt for nt in nucleotide_counts.keys() if nt not in standard_nucleotides]
    
    return {
        'counts': nucleotide_counts,
        'percentages': percentages,
        'gc_content': gc_content,
        'length': total_count,
        'non_standard': non_standard
    }

def check_sequence_quality(analysis_result, min_gc=25, max_gc=75):
    """Check if a sequence meets quality criteria"""
    issues = []
    
    # Check GC content
    gc_content = analysis_result['gc_content']
    if gc_content < min_gc:
        issues.append(f"Low GC content: {gc_content:.2f}% (below {min_gc}%)")
    elif gc_content > max_gc:
        issues.append(f"High GC content: {gc_content:.2f}% (above {max_gc}%)")
    
    # Check for non-standard nucleotides
    if analysis_result['non_standard']:
        issues.append(f"Contains non-standard nucleotides: {', '.join(analysis_result['non_standard'])}")
    
    # Check for extreme nucleotide bias
    for nt, percentage in analysis_result['percentages'].items():
        if nt in ['A', 'T', 'G', 'C'] and percentage > 40:
            issues.append(f"Possible nucleotide bias: {nt} appears at {percentage:.2f}%")
    
    # Check sequence length
    if analysis_result['length'] < 100:
        issues.append(f"Sequence is very short: {analysis_result['length']} bp")
    
    return issues

def plot_gc_distribution(gc_values, output_file="gc_content_distribution.png"):
    """Plot the distribution of GC content across sequences"""
    plt.figure(figsize=(10, 6))
    plt.hist(gc_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(gc_values), color='red', linestyle='dashed', linewidth=1, 
                label=f'Mean: {np.mean(gc_values):.2f}%')
    plt.axvline(np.median(gc_values), color='green', linestyle='dashed', linewidth=1, 
                label=f'Median: {np.median(gc_values):.2f}%')
    
    plt.title('GC Content Distribution')
    plt.xlabel('GC Content (%)')
    plt.ylabel('Number of Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    print(f"GC distribution plot saved as {output_file}")

def plot_sequence_lengths(lengths, output_file="sequence_length_distribution.png"):
    """Plot the distribution of sequence lengths"""
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(np.mean(lengths), color='red', linestyle='dashed', linewidth=1, 
                label=f'Mean: {np.mean(lengths):.2f} bp')
    plt.axvline(np.median(lengths), color='blue', linestyle='dashed', linewidth=1, 
                label=f'Median: {np.median(lengths):.2f} bp')
    
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length (bp)')
    plt.ylabel('Number of Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    print(f"Sequence length distribution plot saved as {output_file}")

def analyze_genbank_file(genbank_file, min_gc=25, max_gc=75, output_dir="./analysis_results"):
    """Analyze all sequences in a GenBank file for quality and GC content"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing GenBank file: {genbank_file}")
    
    # Parse the GenBank file
    records = list(SeqIO.parse(genbank_file, "genbank"))
    print(f"Found {len(records)} sequence records in the file")
    
    if len(records) == 0:
        print("Error: No sequences found in the GenBank file")
        return
    
    # Analyze each sequence
    all_results = []
    gc_values = []
    sequence_lengths = []
    sequences_with_issues = []
    
    for i, record in enumerate(records):
        print(f"\nAnalyzing sequence {i+1}/{len(records)}: {record.id} - {record.description[:50]}...")
        
        # Get sequence
        sequence = record.seq
        
        # Check if sequence is defined
        try:
            sequence_str = str(sequence).upper()
            sequence_length = len(sequence)
            sequence_lengths.append(sequence_length)
            
            # Calculate GC content manually
            gc_count = sequence_str.count('G') + sequence_str.count('C')
            gc_content = (gc_count / sequence_length) * 100 if sequence_length > 0 else 0
        except Bio.Seq.UndefinedSequenceError:
            print(f"  Warning: Sequence {record.id} has undefined content, skipping analysis")
            sequences_with_issues.append((record.id, ["Undefined sequence content"]))
            continue
        gc_values.append(gc_content)
        
        # Detailed analysis
        analysis = analyze_sequence(sequence)
        
        # Check for quality issues
        issues = check_sequence_quality(analysis, min_gc, max_gc)
        
        if issues:
            sequences_with_issues.append((record.id, issues))
            print(f"  Issues found with sequence {record.id}:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  Sequence {record.id} passed all quality checks")
        
        print(f"  Length: {len(sequence)} bp")
        print(f"  GC content: {gc_content:.2f}%")
        
        all_results.append({
            'id': record.id,
            'description': record.description,
            'length': len(sequence),
            'gc_content': gc_content,
            'analysis': analysis,
            'issues': issues
        })
    
    # Generate summary report
    with open(os.path.join(output_dir, "genbank_analysis_report.txt"), "w") as report_file:
        report_file.write(f"GenBank Analysis Report for {genbank_file}\n")
        report_file.write(f"{'='*50}\n\n")
        report_file.write(f"Total sequences analyzed: {len(records)}\n")
        report_file.write(f"Sequences with issues: {len(sequences_with_issues)}\n\n")
        
        report_file.write("Summary Statistics:\n")
        report_file.write(f"  Average sequence length: {np.mean(sequence_lengths):.2f} bp\n")
        report_file.write(f"  Median sequence length: {np.median(sequence_lengths):.2f} bp\n")
        report_file.write(f"  Average GC content: {np.mean(gc_values):.2f}%\n")
        report_file.write(f"  Median GC content: {np.median(gc_values):.2f}%\n\n")
        
        if sequences_with_issues:
            report_file.write("Sequences with Issues:\n")
            for seq_id, issues in sequences_with_issues:
                report_file.write(f"  {seq_id}:\n")
                for issue in issues:
                    report_file.write(f"    - {issue}\n")
                report_file.write("\n")
        else:
            report_file.write("All sequences passed quality checks.\n")
    
    print(f"\nAnalysis report saved to {os.path.join(output_dir, 'genbank_analysis_report.txt')}")
    
    # Generate plots
    plot_gc_distribution(gc_values, os.path.join(output_dir, "gc_content_distribution.png"))
    plot_sequence_lengths(sequence_lengths, os.path.join(output_dir, "sequence_length_distribution.png"))
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Analyze GenBank file for sequence quality and GC content')
    parser.add_argument('--genbank', type=str, default='2Mar2025_phages_downloaded_from_genbank.gb', 
                        help='Path to the GenBank file')
    parser.add_argument('--min-gc', type=float, default=25.0, 
                        help='Minimum acceptable GC content percentage')
    parser.add_argument('--max-gc', type=float, default=75.0, 
                        help='Maximum acceptable GC content percentage')
    parser.add_argument('--output-dir', type=str, default='./analysis_results', 
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    analyze_genbank_file(args.genbank, args.min_gc, args.max_gc, args.output_dir)

if __name__ == "__main__":
    main()