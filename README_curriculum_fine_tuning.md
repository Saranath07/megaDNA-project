# Curriculum Fine-Tuning for DNA Sequence Generation

This README explains how to use the curriculum fine-tuning approach to properly train DNA sequence generation models. This approach addresses the issue of models generating sequences with poor nucleotide distribution (e.g., mostly A's) by using curriculum learning and a custom nucleotide-balanced loss function.

## Problem Addressed

The original fine-tuning process had several limitations that led to poor quality generated sequences:

1. **Nucleotide Imbalance**: The model tends to generate sequences with mostly A's despite training on data with good GC content
2. **Insufficient Training**: Training for only a few epochs is not enough for the model to learn proper DNA patterns
3. **Lack of Regularization**: No specific mechanisms to encourage balanced nucleotide distribution
4. **Fixed Sequence Length**: No gradual increase in sequence complexity during training

## Solution: Curriculum Learning with Balanced Loss

Our solution combines two powerful techniques:

1. **Custom Nucleotide-Balanced Loss Function**: Penalizes imbalanced nucleotide distributions
2. **Curriculum Learning**: Gradually increases sequence complexity during training

## Files in this Package

- `custom_dna_loss.py`: Contains the custom loss function and curriculum trainer
- `curriculum_fine_tuning.py`: Main script for fine-tuning with curriculum learning
- `README_curriculum_fine_tuning.md`: This documentation file

## How the Custom Loss Function Works

The `NucleotideBalancedLoss` class combines standard cross-entropy loss with penalties for:

1. **GC Content Deviation**: Penalizes deviation from target GC content
2. **Complementary Pair Imbalance**: Penalizes imbalance between A-T and G-C pairs
3. **Overall Nucleotide Distribution**: Encourages even distribution of all nucleotides

```python
total_loss = (1 - balance_weight) * cross_entropy_loss + balance_weight * balance_penalty
```

Where `balance_penalty` combines penalties for GC content deviation and complementary pair imbalance.

## How Curriculum Learning Works

The `CurriculumDNATrainer` class implements curriculum learning by:

1. **Starting with Short Sequences**: Initially trains on short sequences (50 bp)
2. **Gradually Increasing Length**: Every 5 epochs, increases sequence length by 50 bp
3. **Adjusting Balance Weight**: Gradually decreases the weight of the balance penalty
4. **Monitoring Progress**: Tracks nucleotide distribution throughout training

## Usage

```bash
python curriculum_fine_tuning.py [options]
```

### Command-line Options

- `--model`: Path to pre-trained model (default: 'notebook/megaDNA_phage_145M.pt')
- `--data`: Path to GenBank data file (default: '2Mar2025_phages_downloaded_from_genbank.gb')
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 8)
- `--max-sequences`: Maximum number of sequences to use (default: 1000)
- `--seed`: Seed sequence for generation (default: 'ATGC')
- `--target-gc`: Target GC content as a ratio (default: 0.5)

### Example

```bash
# Fine-tune with default parameters
python curriculum_fine_tuning.py

# Fine-tune with custom parameters
python curriculum_fine_tuning.py --epochs 100 --batch-size 16 --target-gc 0.45 --seed "GCTA"
```

## Output

The script creates a `curriculum_training` directory with the following structure:

```
curriculum_training/
├── plots/                  # Loss curves, nucleotide progression, etc.
├── checkpoints/            # Model checkpoints saved every 10 epochs
└── sequences/              # Generated sequences during training
```

The final fine-tuned model is saved as `curriculum_fine_tuned_model.pt` in the root directory.

## Monitoring Training Progress

During training, the script will:

1. Print loss values for each epoch
2. Generate and analyze sequences every 5 epochs
3. Plot nucleotide distribution and progression
4. Save model checkpoints every 10 epochs

## Why This Approach Works Better

### 1. Addressing the "Only A's" Problem

The custom loss function directly penalizes the model for generating imbalanced nucleotide distributions. By adding a specific penalty term for deviations from the target GC content and complementary pair imbalance, the model is encouraged to generate more balanced sequences.

### 2. Curriculum Learning Benefits

Starting with shorter sequences and gradually increasing complexity helps the model learn basic patterns first before tackling longer-range dependencies. This approach:

- Makes initial training easier and more stable
- Allows the model to build up complexity gradually
- Prevents the model from falling into local minima early in training

### 3. Proper Regularization

The combination of:
- Weight decay in the optimizer
- Gradient clipping
- Custom balance penalties
- Learning rate scheduling

All work together to prevent overfitting and ensure stable training.

## Troubleshooting

If you encounter issues:

1. **Out of Memory**: Reduce batch size or max_sequences
2. **Slow Training**: Reduce the number of epochs or use a smaller subset of data
3. **Still Generating Imbalanced Sequences**: 
   - Increase the balance_weight parameter in the loss function
   - Train for more epochs
   - Try different learning rates

## Next Steps After Fine-Tuning

1. Use `generate_from_fine_tuned.py` to generate sequences with your fine-tuned model
2. Analyze the generated sequences with `analyze_generated_sequences.py`
3. Compare the quality of sequences with those from the original model

## Technical Details

### Custom Loss Function Parameters

- `balance_weight`: Weight for the balance penalty term (0-1)
- `target_gc_content`: Target GC content (0-1)
- `vocab_size`: Size of the vocabulary (typically 6 for DNA: A,T,G,C,**,#)

### Curriculum Parameters

- `current_seq_length`: Starting sequence length (default: 50)
- `max_seq_length`: Maximum sequence length (default: 500)
- Sequence length increases by 50 every 5 epochs
- Balance weight decreases by 0.05 every 10 epochs (minimum: 0.1)