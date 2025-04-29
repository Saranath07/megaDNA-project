from Bio import SeqIO
import torch
from torch.optim import AdamW
import re
from megadna_tokenizer import NucleotideTokenizer, GenomicDataset, tokenize_sequences, generate_sequence, nucleotides
import numpy as np

def load_sequences_from_genbank(file_path):
    sequences = []
    skipped_ids = []  # To keep track of skipped records due to missing sequences
    problematic_records = []  # To track records causing errors
    
    for record in SeqIO.parse(file_path, "genbank"):
        try:
            # Check if the sequence is None or empty
            if not record.seq or len(record.seq) == 0:
                skipped_ids.append(record.id)  # Store the skipped record ID
                print(f"Skipping record with missing or invalid sequence: {record.id}")
            else:
                sequences.append(str(record.seq))  # Store valid sequences as strings
        except Exception as e:
            # Track records that throw errors during sequence handling
            problematic_records.append(record.id)
            print(f"Error processing record {record.id}: {str(e)}")

    return sequences, skipped_ids, problematic_records

def main():
    # Path to your GenBank file
    genbank_file_path = '2Mar2025_phages_downloaded_from_genbank.gb'
    
    # Load the sequences
    print("Loading sequences from GenBank file...")
    sequences, skipped_ids, problematic_records = load_sequences_from_genbank(genbank_file_path)
    
    # Print number of valid sequences, skipped records, and problematic records
    print(f"Loaded {len(sequences)} valid sequences.")
    print(f"Skipped records: {len(skipped_ids)}")
    print(f"Problematic record IDs (with errors): {problematic_records[:10]}")  # Show first 10 problematic records if any
    
    # Tokenize the sequences
    print("Tokenizing sequences...")
    tokenized_sequences = tokenize_sequences(sequences)
    print(f"First tokenized sequence (first 20 tokens): {tokenized_sequences[0][:20]}")
    
    # Initialize the tokenizer
    tokenizer = NucleotideTokenizer(nucleotides)
    
    # Create dataset and dataloader
    print("Creating dataset and dataloader...")
    dataset = GenomicDataset(tokenized_sequences, tokenizer)
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Try to load the model
    try:
        print("Loading model...")
        model_path = "notebook/megaDNA_phage_145M.pt"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model = torch.load(model_path, map_location=torch.device(device))
        model.eval()
        
        # Freeze layers for fine-tuning
        print("Freezing early layers for fine-tuning...")
        
        # Identify model architecture - assuming it's a transformer-based model
        # Typically we freeze embedding layers and early transformer blocks
        trainable_params = []
        frozen_params_count = 0
        total_params_count = 0
        
        # Common pattern: freeze embeddings and first N transformer layers
        for name, param in model.named_parameters():
            total_params_count += 1
            
            # Freeze embedding layers
            if 'embed' in name:
                param.requires_grad = False
                frozen_params_count += 1
                continue
                
            # Freeze early transformer layers (assuming layer numbering in name)
            # This regex looks for layer numbers in the parameter name
            layer_match = re.search(r'layer\.(\d+)', name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                # Freeze first 70% of layers (adjust this threshold as needed)
                # Try to get num_hidden_layers from config, or use a default value
                try:
                    num_hidden_layers = getattr(model.config, 'num_hidden_layers', 12)
                    if layer_num < (num_hidden_layers * 0.7):
                        param.requires_grad = False
                        frozen_params_count += 1
                        continue
                except AttributeError:
                    # If model doesn't have config attribute, use a heuristic approach
                    # Assuming layer numbers start from 0, freeze layers 0-7 (first 8 layers)
                    if layer_num < 8:
                        param.requires_grad = False
                        frozen_params_count += 1
                        continue
            
            # For all other parameters, keep them trainable
            trainable_params.append(param)
        
        print(f"Frozen {frozen_params_count}/{total_params_count} parameters. "
              f"Fine-tuning {total_params_count - frozen_params_count} parameters.")
        
        # Set up the optimizer with only trainable parameters
        optimizer = AdamW(trainable_params, lr=5e-6)
        model.to(device)
        
        # Fine-tuning loop
        print("Starting fine-tuning...")
        num_epochs = 1  # Reduced for testing, increase for actual training
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                inputs = batch.to(device)
                labels = inputs.clone().detach()
                optimizer.zero_grad()
                
                try:
                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    batch_count += 1
                    
                    if batch_count % 10 == 0:
                        print(f"Epoch {epoch + 1}, Batch {batch_count}, Loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Generate a sample sequence
        print("Generating sample sequence...")
        seed_sequence = "ATGCGTACGTAGC"  # Example seed
        generated_seq = generate_sequence(model, tokenizer, seed_sequence, device=device)
        print(f"Generated Sequence (first 100 chars): {generated_seq[:100]}")
        
        # Save the fine-tuned model with frozen layers
        print("Saving model...")
        try:
            # Try Hugging Face save_pretrained method first
            try:
                model.save_pretrained("./fine_tuned_megaDNA")
                tokenizer.save_pretrained("./fine_tuned_megaDNA")
                print("Model and tokenizer saved successfully using save_pretrained.")
            except AttributeError:
                # Fallback to PyTorch save if save_pretrained is not available
                torch.save(model, "./fine_tuned_megaDNA_model.pt")
                try:
                    tokenizer.save_pretrained("./fine_tuned_megaDNA")
                except AttributeError:
                    print("Could not save tokenizer with save_pretrained, it may not be a Hugging Face tokenizer.")
                print("Model saved successfully using torch.save.")
        except Exception as e:
            print(f"Error saving model: {e}")
            
    except Exception as e:
        print(f"Error loading or using model: {e}")
        print("Skipping training and generation steps.")

if __name__ == "__main__":
    main()