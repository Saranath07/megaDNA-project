#!/usr/bin/env python3
"""
Custom loss function for DNA sequence generation that penalizes nucleotide imbalance.
This helps ensure generated sequences have proper GC content and nucleotide distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NucleotideBalancedLoss(nn.Module):
    """
    Custom loss function that combines cross-entropy with a nucleotide balance penalty.
    This encourages the model to generate sequences with balanced nucleotide distributions.
    """
    def __init__(self, balance_weight=0.5, target_gc_content=0.5, vocab_size=6):
        """
        Initialize the loss function.
        
        Args:
            balance_weight: Weight for the balance penalty term (0-1)
            target_gc_content: Target GC content (0-1)
            vocab_size: Size of the vocabulary (typically 6 for DNA: A,T,G,C,**,#)
        """
        super().__init__()
        self.balance_weight = balance_weight
        self.target_gc_content = target_gc_content
        self.vocab_size = vocab_size
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Assuming token IDs: A=1, T=2, G=3, C=4 (adjust if different)
        self.a_id = 1
        self.t_id = 2
        self.g_id = 3
        self.c_id = 4
    
    def forward(self, logits, targets):
        """
        Calculate the combined loss.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            
        Returns:
            Combined loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross-entropy
        ce_logits = logits.view(-1, vocab_size)
        ce_targets = targets.view(-1)
        
        # Calculate standard cross-entropy loss
        ce_loss = self.ce_loss(ce_logits, ce_targets)
        
        # Calculate nucleotide balance penalty
        # Get probabilities for each nucleotide
        probs = F.softmax(logits, dim=-1)
        
        # Extract probabilities for A, T, G, C
        a_probs = probs[:, :, self.a_id].mean()
        t_probs = probs[:, :, self.t_id].mean()
        g_probs = probs[:, :, self.g_id].mean()
        c_probs = probs[:, :, self.c_id].mean()
        
        # Calculate GC content in the predicted distribution
        gc_content = (g_probs + c_probs) / (a_probs + t_probs + g_probs + c_probs + 1e-8)
        
        # Penalize deviation from target GC content
        gc_penalty = (gc_content - self.target_gc_content) ** 2
        
        # Penalize imbalance between complementary pairs (A-T and G-C)
        at_penalty = (a_probs - t_probs) ** 2
        gc_pair_penalty = (g_probs - c_probs) ** 2
        
        # Combine penalties
        balance_penalty = gc_penalty + at_penalty + gc_pair_penalty
        
        # Combine losses
        total_loss = (1 - self.balance_weight) * ce_loss + self.balance_weight * balance_penalty
        
        return total_loss

class CurriculumDNATrainer:
    """
    Implements curriculum learning for DNA sequence generation.
    Gradually increases sequence length and complexity during training.
    """
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize the trainer.
        
        Args:
            model: The DNA sequence generation model
            tokenizer: Tokenizer for DNA sequences
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.current_seq_length = 50  # Start with short sequences
        self.max_seq_length = 500
        self.current_epoch = 0
        
        # Use custom loss function
        self.loss_fn = NucleotideBalancedLoss(balance_weight=0.3)
        
        # Optimizer with lower learning rate for stability
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
    
    def prepare_batch(self, sequences):
        """
        Prepare a batch of sequences for training, applying curriculum constraints.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Tensor of tokenized sequences
        """
        # Truncate sequences according to current curriculum stage
        truncated_seqs = [seq[:self.current_seq_length] for seq in sequences]
        
        # Tokenize sequences
        tokenized = [self.tokenizer.encode(seq) for seq in truncated_seqs]
        
        # Convert to tensor
        batch = torch.tensor(tokenized).to(self.device)
        
        return batch
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader providing batches of sequences
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_seqs in dataloader:
            # Prepare input batch according to curriculum
            inputs = self.prepare_batch(batch_seqs)
            
            # Create targets (shifted inputs for next-token prediction)
            targets = inputs.clone()
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                # Try different ways to get outputs based on model architecture
                try:
                    # For models that return logits directly
                    logits = self.model(inputs)
                    
                    # Calculate loss with custom loss function
                    loss = self.loss_fn(logits[:, :-1, :], targets[:, 1:])
                    
                except TypeError:
                    # For models with different forward signature
                    outputs = self.model(inputs)
                    
                    # Try to get logits from outputs
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        loss = self.loss_fn(logits[:, :-1, :], targets[:, 1:])
                    else:
                        # Fall back to model's own loss calculation
                        outputs = self.model(inputs, labels=targets)
                        loss = outputs.loss
                
                # Backward pass and optimization
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        # Update curriculum for next epoch
        self.update_curriculum()
        
        # Calculate average loss
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        # Update learning rate
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def update_curriculum(self):
        """Update the curriculum parameters based on current epoch"""
        self.current_epoch += 1
        
        # Gradually increase sequence length
        if self.current_epoch % 5 == 0:
            self.current_seq_length = min(
                self.current_seq_length + 50,  # Increase by 50 every 5 epochs
                self.max_seq_length
            )
            print(f"Curriculum updated: sequence length increased to {self.current_seq_length}")
        
        # Gradually decrease balance weight to focus more on accuracy
        if self.current_epoch % 10 == 0:
            self.loss_fn.balance_weight = max(
                self.loss_fn.balance_weight - 0.05,
                0.1  # Minimum balance weight
            )
            print(f"Curriculum updated: balance weight decreased to {self.loss_fn.balance_weight}")

def analyze_nucleotide_balance(model, tokenizer, seed_sequence="ATGC", device='cuda'):
    """
    Analyze the model's tendency to generate balanced nucleotides.
    
    Args:
        model: The DNA sequence generation model
        tokenizer: Tokenizer for DNA sequences
        seed_sequence: Seed sequence to start generation
        device: Device to run on
        
    Returns:
        Dictionary with analysis results
    """
    model.eval()
    
    # Token IDs for nucleotides (adjust if different)
    a_id = 1  # A
    t_id = 2  # T
    g_id = 3  # G
    c_id = 4  # C
    
    # Encode seed sequence
    input_ids = tokenizer.encode(seed_sequence)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    # Get next token predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Get logits for the last position
        if hasattr(outputs, 'logits'):
            logits = outputs.logits[:, -1, :]
        else:
            logits = outputs[:, -1, :]
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)[0]
        
        # Extract probabilities for A, T, G, C
        nucleotide_probs = {
            'A': probs[a_id].item(),
            'T': probs[t_id].item(),
            'G': probs[g_id].item(),
            'C': probs[c_id].item()
        }
        
        # Calculate GC content in predicted distribution
        gc_content = (nucleotide_probs['G'] + nucleotide_probs['C']) / sum(nucleotide_probs.values())
        
        # Calculate balance metrics
        at_balance = 1 - abs(nucleotide_probs['A'] - nucleotide_probs['T']) / (nucleotide_probs['A'] + nucleotide_probs['T'] + 1e-8)
        gc_balance = 1 - abs(nucleotide_probs['G'] - nucleotide_probs['C']) / (nucleotide_probs['G'] + nucleotide_probs['C'] + 1e-8)
        
        return {
            'nucleotide_probs': nucleotide_probs,
            'gc_content': gc_content,
            'at_balance': at_balance,
            'gc_balance': gc_balance
        }

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test custom DNA loss function')
    parser.add_argument('--model', type=str, default='fine_tuned_megaDNA_model.pt',
                        help='Path to model file')
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        from megadna_tokenizer import NucleotideTokenizer, nucleotides
        
        # Add the safe globals for the MEGADNA class
        torch.serialization.add_safe_globals(['megaDNA.megadna.MEGADNA'])
        
        # Load the model with weights_only=False
        model = torch.load(args.model, map_location=torch.device(device), weights_only=False)
        tokenizer = NucleotideTokenizer(nucleotides)
        
        # Analyze nucleotide balance
        print("Analyzing nucleotide balance...")
        analysis = analyze_nucleotide_balance(model, tokenizer, device=device)
        
        print("\nNucleotide probability distribution:")
        for nt, prob in analysis['nucleotide_probs'].items():
            print(f"  {nt}: {prob:.4f}")
        
        print(f"\nPredicted GC content: {analysis['gc_content']:.4f}")
        print(f"A-T balance: {analysis['at_balance']:.4f}")
        print(f"G-C balance: {analysis['gc_balance']:.4f}")
        
        # Create a test batch and loss function
        print("\nTesting custom loss function...")
        loss_fn = NucleotideBalancedLoss(balance_weight=0.5)
        
        # Create dummy data
        batch_size = 2
        seq_len = 10
        vocab_size = 6
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Calculate loss
        loss = loss_fn(logits, targets)
        print(f"Test loss value: {loss.item()}")
        
    except Exception as e:
        print(f"Error: {e}")