#!/usr/bin/env python3
import argparse
import os
import sys
import torch
from .megadna import MEGADNA

def main():
    parser = argparse.ArgumentParser(description='MegaDNA: a long-context generative model of bacteriophage genome')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True, help='Path to the model file (.pt)')
    
    # Generation options
    parser.add_argument('--primer', type=str, default=None, help='Primer sequence to start generation')
    parser.add_argument('--seq_len', type=int, default=1024, help='Length of sequence to generate')
    parser.add_argument('--temperature', type=float, default=0.95, help='Sampling temperature')
    parser.add_argument('--filter_thres', type=float, default=0.0, help='Filter threshold for sampling')
    parser.add_argument('--output', type=str, default=None, help='Output file path (default: stdout)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Load the model
    try:
        print(f"Loading model from {args.model}...")
        model = torch.load(args.model, map_location=torch.device(args.device), weights_only=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Generate sequence
    try:
        print(f"Generating sequence with length {args.seq_len}...")
        seq_tokenized = model.generate(
            prime=args.primer,
            seq_len=args.seq_len,
            temperature=args.temperature,
            filter_thres=args.filter_thres
        )
        
        # Define nucleotides vocabulary
        nucleotides = ['**', 'A', 'T', 'C', 'G', '#']
        
        # Convert tokens to nucleotides
        def token2nucleotide(s):
            return nucleotides[s]
        
        generated_sequence = ''.join(map(token2nucleotide, seq_tokenized.squeeze().cpu().int()))
        
        # Output the generated sequence
        if args.output:
            with open(args.output, 'w') as f:
                f.write(generated_sequence)
            print(f"Generated sequence saved to {args.output}")
        else:
            print(generated_sequence)
            
    except Exception as e:
        print(f"Error generating sequence: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()