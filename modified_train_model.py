import torch
from torch.optim import AdamW
import re
import os
import pickle
import matplotlib.pyplot as plt
from megadna_tokenizer import NucleotideTokenizer, GenomicDataset, tokenize_sequences, generate_sequence, nucleotides
import numpy as np
from torch.serialization import add_safe_globals
from megaDNA.megadna import MEGADNA

def save_sequence_to_fasta(sequence, file_path, header="Generated DNA sequence"):
    """
    Save a DNA sequence to a FASTA format file
    
    Args:
        sequence (str): The DNA sequence to save
        file_path (str): Path to save the FASTA file
        header (str): Header for the FASTA sequence
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(f">{header}\n")
        # Write sequence in lines of 80 characters (standard FASTA format)
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")
    print(f"Sequence saved to {file_path}")

def load_preprocessed_sequences(file_path):
    """
    Load preprocessed sequences from a pickle file
    """
    print(f"Loading preprocessed sequences from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            sequences = pickle.load(f)
        print(f"Successfully loaded {len(sequences)} preprocessed sequences")
        return sequences
    except FileNotFoundError:
        print(f"Error: Preprocessed file {file_path} not found.")
        print("Please run preprocess_genbank.py first to create the preprocessed data.")
        return []
    except Exception as e:
        print(f"Error loading preprocessed sequences: {str(e)}")
        return []

def plot_loss_curve(losses, output_path="training_results/plots/fine_tuning_loss.png"):
    """
    Plot and save the loss curve
    
    Args:
        losses (list): List of loss values
        output_path (str): Path to save the plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o')
    plt.title('Fine-tuning Loss Progression')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Loss curve saved to {output_path}")

def main():
    # Path to preprocessed sequences file
    preprocessed_dir = 'preprocessed_data'
    sequences_file = os.path.join(preprocessed_dir, 'valid_sequences.pkl')
    
    # Define genome sequence length for complete genome generation
    # Typical bacterial genomes range from 1-10 million base pairs
    # Viral genomes are smaller, typically 5,000-200,000 base pairs
    genome_seq_length = 1000  # Set to 100,000 base pairs for complete viral genome
    
    # Check if preprocessed data exists
    if not os.path.exists(sequences_file):
        print(f"Preprocessed data not found at {sequences_file}")
        print("Please run preprocess_genbank.py first to create the preprocessed data.")
        return
    
    # Load the preprocessed sequences
    sequences = load_preprocessed_sequences(sequences_file)
    
    # If no sequences were loaded, exit
    if not sequences:
        print("No sequences were loaded. Exiting.")
        return
    
    print(f"Working with {len(sequences)} preprocessed sequences.")
    
    # Tokenize the sequences
    print("Tokenizing sequences...")
    tokenized_sequences = tokenize_sequences(sequences)
    print(f"First tokenized sequence (first 20 tokens): {tokenized_sequences[0][:20]}")
    
    # Initialize the tokenizer
    tokenizer = NucleotideTokenizer(nucleotides)
    
    # Create dataset and dataloader
    print("Creating dataset and dataloader...")
    dataset = GenomicDataset(tokenized_sequences, tokenizer, max_length=genome_seq_length)
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Try to load the model
   
    print("Loading model...")
    model_path = "notebook/megaDNA_phage_145M.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Add MEGADNA to safe globals list for unpickling
    add_safe_globals([MEGADNA])
    
    model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.eval()
    
    # ===== STEP 1: Generate sequence from pretrained model =====
    print("\n===== PRETRAINED MODEL OUTPUT =====")
    print("Generating sample sequence from pretrained model...")
    seed_sequence = "ATGCGTACGTAGC"  # Example seed
    pretrained_seq = generate_sequence(model, tokenizer, seed_sequence, max_length=genome_seq_length, device=device)
    print(f"Pretrained Model Generated Sequence (first 100 chars): {pretrained_seq[:100]}")
    print(f"Full sequence length: {len(pretrained_seq)}")
    
    # Analyze nucleotide distribution in pretrained output
    nucleotide_counts = {
        'A': pretrained_seq.count('A'),
        'T': pretrained_seq.count('T'),
        'G': pretrained_seq.count('G'),
        'C': pretrained_seq.count('C'),
        'Other': len(pretrained_seq) - (pretrained_seq.count('A') + pretrained_seq.count('T') + 
                                        pretrained_seq.count('G') + pretrained_seq.count('C'))
    }
    print("Pretrained model nucleotide distribution:")
    for nuc, count in nucleotide_counts.items():
        percentage = (count / len(pretrained_seq)) * 100 if len(pretrained_seq) > 0 else 0
        print(f"{nuc}: {count} ({percentage:.2f}%)")
    
    # ===== STEP 2: Fine-tune with progressive layer unfreezing =====
    print("\n===== FINE-TUNING WITH PROGRESSIVE LAYER UNFREEZING =====")
    
    # Create a copy of the model for fine-tuning to preserve the original
    # Use the same safe globals approach for loading the fine-tuning model
    fine_tune_model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    
    # Freeze initial layers but unfreeze later transformer layers
    print("Implementing progressive layer unfreezing strategy...")
    trainable_params = []
    frozen_params_count = 0
    total_params_count = 0
    
    # Freeze all parameters by default
    for name, param in fine_tune_model.named_parameters():
        total_params_count += 1
        param.requires_grad = False
        frozen_params_count += 1
    
    # Implement a more sophisticated layer freezing strategy:
    # 1. Keep initial layers frozen (embeddings and early transformer layers)
    # 2. Unfreeze later transformer layers and output layers
    
    # Define which layers to unfreeze based on layer depth
    for name, param in fine_tune_model.named_parameters():
        # Always unfreeze output layers
        if 'lm_head' in name or 'to_logits' in name:
            param.requires_grad = True
            frozen_params_count -= 1
            trainable_params.append(param)
            print(f"Unfreezing output layer: {name}")
        
        # Unfreeze later transformer layers (e.g., layers 4 and above)
        elif 'transformers' in name:
            # Extract layer number if possible
            layer_match = re.search(r'transformers_(\d+)_layers_(\d+)', name)
            if layer_match:
                transformer_idx = int(layer_match.group(1))
                layer_idx = int(layer_match.group(2))
                
                # Unfreeze later layers in each transformer block
                # Adjust these thresholds based on your model architecture
                if (transformer_idx >= 1) or (transformer_idx == 0 and layer_idx >= 4):
                    param.requires_grad = True
                    frozen_params_count -= 1
                    trainable_params.append(param)
                    print(f"Unfreezing transformer layer: {name}")
        
        # Unfreeze all output projection layers
        elif name.endswith('out.weight') or name.endswith('out.bias'):
            param.requires_grad = True
            frozen_params_count -= 1
            trainable_params.append(param)
            print(f"Unfreezing output projection: {name}")
    
    print(f"Frozen {frozen_params_count}/{total_params_count} parameters. "
            f"Fine-tuning {total_params_count - frozen_params_count} parameters.")
    
    # Set up the optimizer with very low learning rate
    optimizer = AdamW(trainable_params, lr=5e-6)  # Slightly higher learning rate for more parameters
    fine_tune_model.to(device)
    
    # Fine-tuning loop - just a few steps
    print("Starting progressive layer fine-tuning...")
    num_epochs = 10  # Reduced number of epochs for demonstration
    max_batches_per_epoch = 5  # Number of batches to process per epoch
    
    fine_tune_model.train()
    epoch_losses = []  # Track average loss per epoch
    all_losses = []  # Track all individual batch losses
    
    # Create directories for saving epoch-specific sequences
    epoch_seq_dir = os.path.join("training_results", "sequences", "epochs")
    os.makedirs(epoch_seq_dir, exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        total_loss = 0
        batch_count = 0
        epoch_batch_losses = []  # Track losses for this epoch
        
        for batch in train_loader:
            if batch_count >= max_batches_per_epoch:
                break
                
            inputs = batch.to(device)
            optimizer.zero_grad()
            
            try:
                # For MEGADNA model, call forward directly without labels
                # The forward method returns the loss directly
                loss = fine_tune_model(inputs)
                
                # Record the loss
                current_loss = loss.item()
                epoch_batch_losses.append(current_loss)
                all_losses.append(current_loss)
                total_loss += current_loss
                
                loss.backward()
                optimizer.step()
                batch_count += 1
                
                print(f"Epoch {epoch}, Batch {batch_count}/{max_batches_per_epoch}, Loss: {current_loss:.4f}")
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        # Calculate and store average loss for this epoch
        avg_epoch_loss = total_loss / batch_count if batch_count > 0 else 0
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch} complete, Average Loss: {avg_epoch_loss:.4f}")
        
        # Generate and save a sample sequence after each epoch
        fine_tune_model.eval()
        epoch_seq = generate_sequence(fine_tune_model, tokenizer, seed_sequence, max_length=genome_seq_length, device=device)
        
        # Save the epoch's generated sequence to a FASTA file
        save_sequence_to_fasta(
            epoch_seq,
            os.path.join(epoch_seq_dir, f"generated_epoch{epoch}.fasta"),
            header=f"Epoch {epoch} generated sequence (seed: {seed_sequence})"
        )
        
        # Switch back to training mode for next epoch
        fine_tune_model.train()
    
    print(f"\nFine-tuning complete across {num_epochs} epochs")
    
    # Plot and save the loss curves
    plot_loss_curve(all_losses, "training_results/plots/fine_tuning_batch_loss.png")
    
    # Plot epoch average losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig("training_results/plots/epoch_average_loss.png")
    print("Epoch average loss curve saved to training_results/plots/epoch_average_loss.png")
    
    # ===== STEP 3: Generate sequence from fine-tuned model =====
    print("\n===== FINE-TUNED MODEL OUTPUT =====")
    fine_tune_model.eval()
    print("Generating sample sequence from fine-tuned model...")
    # Use the same seed for fair comparison
    finetuned_seq = generate_sequence(fine_tune_model, tokenizer, seed_sequence, max_length=genome_seq_length, device=device)
    print(f"Fine-tuned Model Generated Sequence (first 100 chars): {finetuned_seq[:100]}")
    print(f"Full sequence length: {len(finetuned_seq)}")
    
    # Save the generated sequences to FASTA files
    output_dir = "training_results/sequences"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pretrained model sequence
    save_sequence_to_fasta(
        pretrained_seq,
        os.path.join(output_dir, "pretrained_model_sequence.fasta"),
        header=f"Pretrained model generated sequence (seed: {seed_sequence})"
    )
    
    # Save fine-tuned model sequence
    save_sequence_to_fasta(
        finetuned_seq,
        os.path.join(output_dir, "fine_tuned_model_sequence.fasta"),
        header=f"Fine-tuned model generated sequence (seed: {seed_sequence})"
    )
    
    # Analyze nucleotide distribution in fine-tuned output
    nucleotide_counts = {
        'A': finetuned_seq.count('A'),
        'T': finetuned_seq.count('T'),
        'G': finetuned_seq.count('G'),
        'C': finetuned_seq.count('C'),
        'Other': len(finetuned_seq) - (finetuned_seq.count('A') + finetuned_seq.count('T') + 
                                        finetuned_seq.count('G') + finetuned_seq.count('C'))
    }
    print("Fine-tuned model nucleotide distribution:")
    for nuc, count in nucleotide_counts.items():
        percentage = (count / len(finetuned_seq)) * 100 if len(finetuned_seq) > 0 else 0
        print(f"{nuc}: {count} ({percentage:.2f}%)")
    
    # ===== STEP 4: Compare the outputs =====
    print("\n===== COMPARISON =====")
    print("Comparing pretrained vs fine-tuned model outputs:")
    
    # Calculate similarity
    min_len = min(len(pretrained_seq), len(finetuned_seq))
    matching_positions = sum(1 for i in range(min_len) if pretrained_seq[i] == finetuned_seq[i])
    similarity = (matching_positions / min_len) * 100 if min_len > 0 else 0
    
    print(f"Sequence similarity: {similarity:.2f}%")
    print(f"Pretrained sequence starts with: {pretrained_seq[:50]}")
    print(f"Fine-tuned sequence starts with: {finetuned_seq[:50]}")
    
    # Check if fine-tuned model is generating mostly A's
    a_percentage = (finetuned_seq.count('A') / len(finetuned_seq)) * 100 if len(finetuned_seq) > 0 else 0
    if a_percentage > 70:
        print(f"WARNING: Fine-tuned model is generating mostly A's ({a_percentage:.2f}%)")
        print("This suggests the fine-tuning process may still be too aggressive.")
        print("Consider further reducing learning rate or trainable parameters.")
    
    # Save the fine-tuned model
    print("\nSaving fine-tuned model...")
    try:
        # Save the model
        torch.save(fine_tune_model, "./progressive_fine_tuned_megaDNA_model.pt")
        print("Model saved successfully.")
        
        # Save the loss values for future reference
        loss_data = {
            'batch_losses': all_losses,
            'epoch_losses': epoch_losses
        }
        loss_data_path = "training_results/loss_values.pkl"
        with open(loss_data_path, 'wb') as f:
            pickle.dump(loss_data, f)
        print(f"Loss values saved to {loss_data_path}")
    except Exception as e:
        print(f"Error saving model or data: {e}")

if __name__ == "__main__":
    main()