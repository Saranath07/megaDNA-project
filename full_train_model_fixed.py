from Bio import SeqIO
import torch
from transformers import AdamW
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
    
    # For testing purposes, limit to a small subset
    test_mode = True
    if test_mode:
        print("Running in test mode with limited sequences...")
        sequences = sequences[:10]  # Use only first 10 sequences for testing
    
    # Tokenize the sequences
    print("Tokenizing sequences...")
    tokenized_sequences = tokenize_sequences(sequences)
    print(f"First tokenized sequence (first 20 tokens): {tokenized_sequences[0][:20]}")
    
    # Initialize the tokenizer
    tokenizer = NucleotideTokenizer(nucleotides)
    
    # Create dataset and dataloader
    print("Creating dataset and dataloader...")
    dataset = GenomicDataset(tokenized_sequences, tokenizer)
    batch_size = 2 if test_mode else 8
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Try to load the model
    try:
        print("Loading model...")
        model_path = "notebook/megaDNA_phage_145M.pt"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Explicitly set weights_only=False to address the PyTorch 2.6 change
        print("Loading model with weights_only=False...")
        model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        model.eval()
        
        # Set up the optimizer
        optimizer = AdamW(model.parameters(), lr=5e-6)
        model.to(device)
        
        # Fine-tuning loop
        print("Starting training...")
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
        
        # Save the fine-tuned model
        print("Saving model...")
        try:
            model.save_pretrained("./fine_tuned_megaDNA")
            tokenizer.save_pretrained("./fine_tuned_megaDNA")
            print("Model and tokenizer saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")
            
    except Exception as e:
        print(f"Error loading or using model: {e}")
        print("Skipping training and generation steps.")

if __name__ == "__main__":
    main()