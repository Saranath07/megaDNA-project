#!/usr/bin/env python3
"""
Improved fine-tuning script for megaDNA model with:
- Loss tracking and plotting
- Layer-wise weight analysis
- Attention mapping visualization
- Sequence quality monitoring
- Proper hyperparameter tuning
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import re
from Bio import SeqIO
from collections import Counter
import seaborn as sns
from tqdm import tqdm
from megadna_tokenizer import NucleotideTokenizer, GenomicDataset, tokenize_sequences, nucleotides

# Create output directories
os.makedirs("training_results", exist_ok=True)
os.makedirs("training_results/plots", exist_ok=True)
os.makedirs("training_results/weights", exist_ok=True)
os.makedirs("training_results/attention", exist_ok=True)
os.makedirs("training_results/sequences", exist_ok=True)

def load_sequences_from_genbank(file_path, max_sequences=None, min_length=100, max_length=10000):
    """
    Load sequences from GenBank file with filtering by length
    
    Args:
        file_path: Path to GenBank file
        max_sequences: Maximum number of sequences to load (None for all)
        min_length: Minimum sequence length to include
        max_length: Maximum sequence length to include
    
    Returns:
        List of sequences, skipped IDs, and problematic records
    """
    sequences = []
    skipped_ids = []
    problematic_records = []
    
    print(f"Loading sequences from {file_path}...")
    for record in SeqIO.parse(file_path, "genbank"):
        try:
            # Check if the sequence is None or empty
            if not record.seq or len(record.seq) == 0:
                skipped_ids.append(record.id)
                continue
                
            # Check sequence length
            if len(record.seq) < min_length or len(record.seq) > max_length:
                skipped_ids.append(record.id)
                continue
                
            # Check for non-standard nucleotides
            seq_str = str(record.seq).upper()
            if any(nt not in "ATGCN" for nt in seq_str):
                skipped_ids.append(record.id)
                continue
                
            sequences.append(seq_str)
            
            # Break if we've reached the maximum number of sequences
            if max_sequences and len(sequences) >= max_sequences:
                break
                
        except Exception as e:
            problematic_records.append(record.id)
            print(f"Error processing record {record.id}: {str(e)}")
    
    return sequences, skipped_ids, problematic_records

def analyze_sequence(sequence):
    """Analyze nucleotide composition of a DNA sequence"""
    nucleotide_counts = Counter(sequence.upper())
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

def plot_loss_curve(train_losses, val_losses=None, save_path="training_results/plots/loss_curve.png"):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='orange')
    
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def plot_layer_weights(model, epoch, save_dir="training_results/weights"):
    """Visualize layer weights distribution"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all weight matrices
    weight_matrices = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:  # Only include weight matrices, not biases
            weight_matrices[name] = param.detach().cpu().numpy()
    
    # Plot histograms for each layer's weights
    for name, weights in weight_matrices.items():
        plt.figure(figsize=(10, 6))
        
        # Flatten weights for histogram
        flat_weights = weights.flatten()
        
        # Plot histogram
        plt.hist(flat_weights, bins=50, alpha=0.7)
        plt.title(f'Weight Distribution - {name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        
        # Add statistics
        plt.axvline(np.mean(flat_weights), color='r', linestyle='dashed', linewidth=1, 
                   label=f'Mean: {np.mean(flat_weights):.4f}')
        plt.axvline(np.median(flat_weights), color='g', linestyle='dashed', linewidth=1,
                   label=f'Median: {np.median(flat_weights):.4f}')
        
        # Add more statistics as text
        stats_text = f"Std: {np.std(flat_weights):.4f}\n"
        stats_text += f"Min: {np.min(flat_weights):.4f}\n"
        stats_text += f"Max: {np.max(flat_weights):.4f}"
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        safe_name = name.replace('.', '_').replace('/', '_')
        save_path = os.path.join(save_dir, f"weights_epoch{epoch}_{safe_name}.png")
        plt.savefig(save_path)
        plt.close()
    
    print(f"Layer weight visualizations for epoch {epoch} saved to {save_dir}")

def visualize_attention(model, tokenizer, input_text, save_path="training_results/attention/attention_map.png"):
    """Visualize attention patterns in the model"""
    # This function needs to be adapted to your specific model architecture
    # Here's a generic implementation that works with transformer models
    
    try:
        # Tokenize input
        input_ids = tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids]).to(next(model.parameters()).device)
        
        # Get attention weights
        model.eval()
        with torch.no_grad():
            # This part depends on your model architecture
            # Try different ways to get attention weights based on model architecture
            try:
                # For models with output_attentions parameter
                outputs = model(input_tensor, output_attentions=True)
                attention = outputs.attentions
            except TypeError:
                print("Model doesn't support output_attentions parameter")
                return
            
            if attention is None:
                print("Model doesn't provide attention weights. Skipping attention visualization.")
                return
            
            # Get the last layer's attention weights
            last_layer_attention = attention[-1].cpu().numpy()
            
            # Average over attention heads
            avg_attention = np.mean(last_layer_attention, axis=1)[0]  # Shape: [seq_len, seq_len]
            
            # Plot attention heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_attention, cmap='viridis')
            plt.title('Attention Map')
            plt.xlabel('Token Position (Target)')
            plt.ylabel('Token Position (Source)')
            plt.savefig(save_path)
            plt.close()
            print(f"Attention map saved to {save_path}")
    except Exception as e:
        print(f"Error visualizing attention: {e}")
        print("Skipping attention visualization.")

def generate_and_analyze_sequence(model, tokenizer, seed_sequence="ATGC", seq_length=500, 
                                 temperature=0.8, epoch=0, device='cpu'):
    """Generate a sequence and analyze its properties"""
    model.eval()
    
    # Encode the seed sequence
    encoded_seed = tokenizer.encode(seed_sequence)
    seed_tensor = torch.tensor([encoded_seed], device=device)
    
    # Generate sequence
    with torch.no_grad():
        try:
            generated_tensor = model.generate(
                prime=seed_tensor,
                seq_len=seq_length,
                filter_thres=0.9,
                temperature=temperature
            )
            
            # Decode the generated sequence
            generated_seq = tokenizer.decode(generated_tensor[0].tolist())
            
            # Analyze the sequence
            analysis = analyze_sequence(generated_seq)
            
            # Save the sequence
            save_path = f"training_results/sequences/generated_epoch{epoch}.fasta"
            with open(save_path, 'w') as f:
                f.write(f">generated_epoch{epoch}_seed_{seed_sequence}_temp_{temperature}\n")
                # Write sequence with line breaks every 80 characters
                for i in range(0, len(generated_seq), 80):
                    f.write(f"{generated_seq[i:i+80]}\n")
            
            # Plot nucleotide distribution
            nucleotides = ['A', 'T', 'G', 'C']
            counts = [analysis['counts'].get(nt, 0) for nt in nucleotides]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(nucleotides, counts)
            
            # Add percentage labels on top of bars
            for i, bar in enumerate(bars):
                percentage = analysis['percentages'].get(nucleotides[i], 0)
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{percentage:.1f}%', ha='center')
            
            plt.title(f'Nucleotide Distribution - Epoch {epoch}')
            plt.xlabel('Nucleotide')
            plt.ylabel('Count')
            plt.savefig(f"training_results/plots/nucleotide_dist_epoch{epoch}.png")
            plt.close()
            
            return generated_seq, analysis
        except Exception as e:
            print(f"Error generating sequence: {e}")
            return None, None

def fine_tune_model(model_path, sequences, num_epochs=50, batch_size=8, learning_rate=1e-5, 
                   val_split=0.1, patience=5, seed_sequence="ATGC"):
    """
    Fine-tune the model with improved monitoring and visualization
    
    Args:
        model_path: Path to the pre-trained model
        sequences: List of sequences for training
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        val_split: Fraction of data to use for validation
        patience: Number of epochs to wait for improvement before early stopping
        seed_sequence: Seed sequence for generation during training
    
    Returns:
        Fine-tuned model and training history
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = NucleotideTokenizer(nucleotides)
    
    # Split data into training and validation sets
    val_size = int(len(sequences) * val_split)
    train_sequences = sequences[:-val_size] if val_size > 0 else sequences
    val_sequences = sequences[-val_size:] if val_size > 0 else []
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # Tokenize sequences
    print("Tokenizing sequences...")
    train_tokenized = tokenize_sequences(train_sequences)
    val_tokenized = tokenize_sequences(val_sequences) if val_sequences else []
    
    # Create datasets and dataloaders
    train_dataset = GenomicDataset(train_tokenized, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_sequences:
        val_dataset = GenomicDataset(val_tokenized, tokenizer)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        model.to(device)
        model.train()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    # Set up optimizer with weight decay for regularization
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'gc_content': [],
        'learning_rates': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model = None
    no_improvement = 0
    
    # Training loop
    print(f"Starting fine-tuning for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs = batch.to(device)
            labels = inputs.clone().detach()
            
            # Forward pass
            optimizer.zero_grad()
            try:
                # Try different ways to compute loss based on model architecture
                try:
                    # For models that accept labels and return loss
                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss
                except TypeError:
                    # For models that don't accept labels directly
                    outputs = model(inputs)
                    
                    # Try to get logits from outputs
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        # Assume outputs are logits directly
                        logits = outputs
                    
                    # Compute loss manually
                    # Shift logits and labels for next-token prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    
                    # Create a simple cross-entropy loss
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))
                
                # Backward pass and optimization
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_sequences:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    inputs = batch.to(device)
                    labels = inputs.clone().detach()
                    
                    try:
                        outputs = model(inputs, labels=labels)
                        loss = outputs.loss
                        val_loss += loss.item()
                        val_batches += 1
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue
            
            # Calculate average validation loss
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            history['val_loss'].append(avg_val_loss)
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model.state_dict().copy()
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Restore best model
                    model.load_state_dict(best_model)
                    break
        
        # Store current learning rate
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Generate and analyze a sequence every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            generated_seq, analysis = generate_and_analyze_sequence(
                model, tokenizer, seed_sequence=seed_sequence, epoch=epoch+1, device=device
            )
            
            if analysis:
                history['gc_content'].append(analysis['gc_content'])
                print(f"Epoch {epoch+1} - Generated sequence GC content: {analysis['gc_content']:.2f}%")
            
            # Visualize layer weights
            plot_layer_weights(model, epoch+1)
            
            # Visualize attention (if applicable)
            visualize_attention(model, tokenizer, seed_sequence, 
                               save_path=f"training_results/attention/attention_map_epoch{epoch+1}.png")
        
        # Plot loss curve after each epoch
        plot_loss_curve(history['train_loss'], history['val_loss'] if val_sequences else None)
        
        # Plot GC content progression
        if history['gc_content']:
            plt.figure(figsize=(10, 6))
            # Create x-axis values based on when we actually recorded GC content
            gc_epochs = []
            for i in range(num_epochs):
                if i % 5 == 0 or i == 0 or i == num_epochs - 1:
                    gc_epochs.append(i+1)
            
            # Make sure we have the right number of x values for our y values
            if len(gc_epochs) == len(history['gc_content']):
                plt.plot(gc_epochs, history['gc_content'], marker='o')
                plt.title('GC Content Progression During Training')
                plt.xlabel('Epoch')
                plt.ylabel('GC Content (%)')
                plt.grid(True, alpha=0.3)
                plt.savefig("training_results/plots/gc_content_progression.png")
                plt.close()
            else:
                print(f"Warning: GC content data points ({len(history['gc_content'])}) don't match expected epochs ({len(gc_epochs)})")
        
        # Plot learning rate progression
        plt.figure(figsize=(10, 6))
        plt.plot(history['learning_rates'], marker='o')
        plt.title('Learning Rate Progression')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig("training_results/plots/learning_rate_progression.png")
        plt.close()
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}" + 
              (f" - Val Loss: {avg_val_loss:.4f}" if val_sequences else "") +
              f" - Time: {epoch_time:.2f}s")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model, f"training_results/model_checkpoint_epoch{epoch+1}.pt")
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save the final model
    print("Saving final fine-tuned model...")
    torch.save(model, "fine_tuned_megaDNA_model.pt")
    
    # Generate a final sequence with the fine-tuned model
    print("Generating final sequence with fine-tuned model...")
    final_seq, final_analysis = generate_and_analyze_sequence(
        model, tokenizer, seed_sequence=seed_sequence, epoch="final", device=device, seq_length=1000
    )
    
    if final_analysis:
        print(f"Final generated sequence stats:")
        print(f"  Length: {final_analysis['length']} bp")
        print(f"  GC content: {final_analysis['gc_content']:.2f}%")
        print(f"  Nucleotide percentages: A={final_analysis['percentages'].get('A', 0):.2f}%, "
              f"T={final_analysis['percentages'].get('T', 0):.2f}%, "
              f"G={final_analysis['percentages'].get('G', 0):.2f}%, "
              f"C={final_analysis['percentages'].get('C', 0):.2f}%")
    
    return model, history

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Improved fine-tuning for megaDNA model')
    parser.add_argument('--model', type=str, default='notebook/megaDNA_phage_145M.pt',
                        help='Path to pre-trained model')
    parser.add_argument('--data', type=str, default='2Mar2025_phages_downloaded_from_genbank.gb',
                        help='Path to GenBank data file')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--max-sequences', type=int, default=1000,
                        help='Maximum number of sequences to use (None for all)')
    parser.add_argument('--seed', type=str, default='ATGC',
                        help='Seed sequence for generation')
    args = parser.parse_args()
    
    # Load sequences from GenBank file
    sequences, skipped_ids, problematic_records = load_sequences_from_genbank(
        args.data, max_sequences=args.max_sequences
    )
    
    print(f"Loaded {len(sequences)} valid sequences")
    print(f"Skipped {len(skipped_ids)} sequences")
    print(f"Encountered {len(problematic_records)} problematic records")
    
    if len(sequences) == 0:
        print("No valid sequences found. Exiting.")
        return
    
    # Analyze training data
    print("Analyzing training data...")
    gc_contents = []
    sequence_lengths = []
    
    for seq in sequences:
        analysis = analyze_sequence(seq)
        gc_contents.append(analysis['gc_content'])
        sequence_lengths.append(analysis['length'])
    
    # Plot GC content distribution of training data
    plt.figure(figsize=(10, 6))
    plt.hist(gc_contents, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.mean(gc_contents), color='red', linestyle='dashed', linewidth=1,
               label=f'Mean: {np.mean(gc_contents):.2f}%')
    plt.title('GC Content Distribution in Training Data')
    plt.xlabel('GC Content (%)')
    plt.ylabel('Number of Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("training_results/plots/training_data_gc_content.png")
    plt.close()
    
    # Plot sequence length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=20, alpha=0.7, color='green')
    plt.axvline(np.mean(sequence_lengths), color='red', linestyle='dashed', linewidth=1,
               label=f'Mean: {np.mean(sequence_lengths):.2f} bp')
    plt.title('Sequence Length Distribution in Training Data')
    plt.xlabel('Sequence Length (bp)')
    plt.ylabel('Number of Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("training_results/plots/training_data_length.png")
    plt.close()
    
    print(f"Training data analysis:")
    print(f"  Average GC content: {np.mean(gc_contents):.2f}%")
    print(f"  Average sequence length: {np.mean(sequence_lengths):.2f} bp")
    
    # Fine-tune the model
    model, history = fine_tune_model(
        model_path=args.model,
        sequences=sequences,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed_sequence=args.seed
    )
    
    if model is not None:
        print("Fine-tuning completed successfully!")
        print("Final model saved as fine_tuned_megaDNA_model.pt")
        print("Training visualizations saved in training_results/ directory")
    else:
        print("Fine-tuning failed.")

if __name__ == "__main__":
    main()