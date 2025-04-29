# MegaDNA Sequence Generation

This repository contains scripts for generating DNA sequences using a fine-tuned MegaDNA model. The model is trained to generate DNA sequences based on seed sequences.

## Available Scripts

### 1. Basic Sequence Generation

`generate_from_fine_tuned.py` - Generate DNA sequences using the fine-tuned model with various parameters.

```bash
python generate_from_fine_tuned.py [options]
```

Options:
- `--model`: Path to the fine-tuned model (default: ./fine_tuned_megaDNA_model.pt)
- `--seed`: Seed DNA sequence to start generation (default: ATGC)
- `--length`: Length of sequence to generate (default: 1000)
- `--temp`: Temperature for generation (0.0-1.0, lower = more deterministic) (default: 0.8)
- `--num`: Number of sequences to generate (default: 1)
- `--output`: Output file prefix (default: "generated")

Example:
```bash
# Generate 3 sequences of length 500 with seed TAGC
python generate_from_fine_tuned.py --seed TAGC --num 3 --length 500 --output my_sequences
```

### 2. Sequence Analysis

`analyze_generated_sequences.py` - Generate and analyze DNA sequences, including nucleotide composition, GC content, and repeating patterns.

```bash
python analyze_generated_sequences.py [options]
```

Options:
- `--model`: Path to the fine-tuned model (default: ./fine_tuned_megaDNA_model.pt)
- `--seeds`: Seed sequences to use (can provide multiple) (default: ATGC GCTA TAGC)
- `--length`: Length of sequences to generate (default: 500)
- `--temp`: Temperature for generation (default: 0.8)
- `--output`: Output FASTA file (default: analyzed_sequences.fasta)
- `--analyze`: Perform analysis on generated sequences (flag)

Example:
```bash
# Generate and analyze sequences with different seeds
python analyze_generated_sequences.py --analyze --length 300 --seeds ATGC GCTA TAGC
```

### 3. Filtered Sequence Generation

`generate_filtered_sequences.py` - Generate DNA sequences that meet specific criteria such as GC content, presence of restriction sites, or open reading frames.

```bash
python generate_filtered_sequences.py [options]
```

Options:
- `--model`: Path to the fine-tuned model (default: ./fine_tuned_megaDNA_model.pt)
- `--seeds`: Seed sequences to use (can provide multiple) (default: ATGC GCTA TAGC)
- `--num`: Number of sequences to generate (default: 5)
- `--length`: Length of sequences to generate (default: 300)
- `--temp`: Temperature for generation (default: 0.9)
- `--gc-min`: Minimum GC content percentage (default: 30)
- `--gc-max`: Maximum GC content percentage (default: 70)
- `--restriction-sites`: Required restriction sites (can provide multiple)
- `--require-orf`: Require sequences to contain an open reading frame (flag)
- `--output`: Output FASTA file (default: filtered_sequences.fasta)
- `--max-attempts`: Maximum number of generation attempts per seed (default: 100)

Example:
```bash
# Generate 5 sequences with GC content between 40-60% that contain EcoRI site
python generate_filtered_sequences.py --gc-min 40 --gc-max 60 --restriction-sites GAATTC --num 5
```

## Understanding the Output

All scripts generate FASTA files containing the generated sequences. The FASTA headers include information about the generation parameters:

```
>generated_dna_sequence_1_seed_ATGC_temp_0.8
ATGCTGAAAAAGAAAAAGAAGAAAAAGAAAAAGAAAAAGAAAAAGAAAAAGAAAAAGAAAAAGAAGAAAAAGAAAAAGAA
...
```

## Observations on Generated Sequences

Based on experiments with the fine-tuned model, we've observed the following patterns:

1. Different seed sequences produce different initial patterns, but then tend to converge to repeating patterns:
   - ATGC seed → Repeating "AAAAAG" patterns
   - GCTA seed → Initial "GCTATCAAAAAAG" followed by repeating "A" nucleotides
   - TAGC seed → Initial "TAGCAAAAATATTATTAAAAAATAAAAAAAATAAAAAAATATTTTTAAAAAATAAAAAAATAAAAAAA" followed by repeating "A" nucleotides

2. The sequences are heavily biased towards adenine (A), with varying amounts of other nucleotides depending on the seed.

3. Changing the temperature parameter affects the randomness of the generation, but the overall patterns remain similar.

## Requirements

- Python 3.6+
- PyTorch
- Biopython (for sequence analysis)
- Matplotlib (for visualization)

## Installation

```bash
pip install torch biopython matplotlib