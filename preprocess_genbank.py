from Bio import SeqIO
import pickle
import os
import time

def load_sequences_from_genbank(file_path):
    sequences = []
    skipped_ids = []  # To keep track of skipped records due to missing sequences
    problematic_records = []  # To track records causing errors
    
    print(f"Processing GenBank file: {file_path}")
    start_time = time.time()
    
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

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    
    return sequences, skipped_ids, problematic_records

def main():
    # Path to your GenBank file
    genbank_file_path = '2Mar2025_phages_downloaded_from_genbank.gb'
    
    # Create preprocessed data directory if it doesn't exist
    preprocessed_dir = 'preprocessed_data'
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)
    
    # Output file paths
    sequences_file = os.path.join(preprocessed_dir, 'valid_sequences.pkl')
    metadata_file = os.path.join(preprocessed_dir, 'preprocessing_metadata.pkl')
    
    # Load the sequences
    print("Loading sequences from GenBank file...")
    sequences, skipped_ids, problematic_records = load_sequences_from_genbank(genbank_file_path)
    
    # Print number of valid sequences, skipped records, and problematic records
    print(f"Loaded {len(sequences)} valid sequences.")
    print(f"Skipped records: {len(skipped_ids)}")
    print(f"Problematic record IDs (with errors): {len(problematic_records)}")
    if problematic_records:
        print(f"First 10 problematic records: {problematic_records[:10]}")
    
    # Save the valid sequences to a file
    print(f"Saving {len(sequences)} valid sequences to {sequences_file}...")
    with open(sequences_file, 'wb') as f:
        pickle.dump(sequences, f)
    
    # Save metadata (skipped and problematic records)
    metadata = {
        'skipped_ids': skipped_ids,
        'problematic_records': problematic_records,
        'total_valid_sequences': len(sequences)
    }
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Preprocessing complete. Data saved to {preprocessed_dir}/")
    print(f"You can now run the training script which will use the preprocessed data.")

if __name__ == "__main__":
    main()