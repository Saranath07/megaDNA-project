# Improved Fine-Tuning for megaDNA Model

This README explains how to use the improved fine-tuning script (`improved_fine_tuning.py`) to properly fine-tune the megaDNA model for generating high-quality DNA sequences with appropriate GC content and nucleotide distribution.

## Problem Addressed

The original fine-tuning process had several limitations that led to poor quality generated sequences (e.g., sequences with only A's):

1. Insufficient training (only 1 epoch)
2. No monitoring of sequence quality during training
3. No visualization of model weights or attention
4. No tracking of loss over epochs
5. No validation to prevent overfitting
6. Suboptimal hyperparameters

## Improvements in the New Script

The improved fine-tuning script includes:

1. **Extended Training**: Trains for multiple epochs (default: 50) instead of just 1
2. **Loss Tracking and Visualization**: Plots loss curves over epochs
3. **Layer-wise Weight Analysis**: Visualizes weight distributions in each layer
4. **Attention Mapping**: Visualizes attention patterns in the model
5. **Sequence Quality Monitoring**: Generates and analyzes sequences during training
6. **Early Stopping**: Prevents overfitting by monitoring validation loss
7. **Learning Rate Scheduling**: Reduces learning rate when progress plateaus
8. **Gradient Clipping**: Prevents exploding gradients
9. **Weight Decay**: Adds regularization to prevent overfitting
10. **Training Data Analysis**: Analyzes GC content and sequence length distribution in training data

## Usage

```bash
python improved_fine_tuning.py [options]
```

### Command-line Options

- `--model`: Path to pre-trained model (default: 'notebook/megaDNA_phage_145M.pt')
- `--data`: Path to GenBank data file (default: '2Mar2025_phages_downloaded_from_genbank.gb')
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 8)
- `--lr`: Learning rate (default: 1e-5)
- `--max-sequences`: Maximum number of sequences to use (default: 1000, use None for all)
- `--seed`: Seed sequence for generation (default: 'ATGC')

### Example

```bash
# Fine-tune with default parameters
python improved_fine_tuning.py

# Fine-tune with custom parameters
python improved_fine_tuning.py --epochs 100 --batch-size 16 --lr 5e-6 --max-sequences 2000 --seed "GCTA"
```

## Output

The script creates a `training_results` directory with the following structure:

```
training_results/
├── plots/                  # Loss curves, GC content progression, etc.
├── weights/                # Layer-wise weight visualizations
├── attention/              # Attention map visualizations
└── sequences/              # Generated sequences during training
```

The final fine-tuned model is saved as `fine_tuned_megaDNA_model.pt` in the root directory.

## Monitoring Training Progress

During training, the script will:

1. Print loss values for each epoch
2. Generate and analyze sequences every 5 epochs
3. Update visualization plots
4. Save model checkpoints every 10 epochs

## Why This Approach Works Better

### 1. Addressing the "Only A's" Problem

The problem of generating sequences with only A's is likely due to:

- **Insufficient Training**: The model needs more epochs to learn the complex patterns in DNA sequences
- **Mode Collapse**: The model falls into a local minimum where generating only A's seems optimal
- **Lack of Regularization**: Without proper regularization, the model can overfit to simple patterns

The improved script addresses these issues by:

- Training for more epochs
- Using weight decay for regularization
- Implementing learning rate scheduling
- Monitoring sequence quality during training

### 2. Proper Sequence Length and Properties

To ensure generated sequences have proper length and properties:

- The script filters training data by length and quality
- It monitors GC content during training
- It uses a higher temperature during generation to increase diversity
- It implements proper tokenization and sequence handling

## Troubleshooting

If you encounter issues:

1. **Out of Memory**: Reduce batch size or max_sequences
2. **Slow Training**: Reduce the number of epochs or use a smaller subset of data
3. **Poor Generation Quality**: Try different learning rates or increase the number of epochs
4. **Model Compatibility**: Ensure the model architecture is compatible with the visualization functions

## Next Steps

After fine-tuning:

1. Use `generate_from_fine_tuned.py` to generate sequences with your fine-tuned model
2. Analyze the generated sequences with `analyze_generated_sequences.py`
3. Compare the quality of sequences with those from the original model