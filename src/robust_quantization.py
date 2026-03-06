import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
import hydra
import os
import random
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'speech_encoder', 'src'))

from speech_encoder import SpeechEncoder
from .models import RobustQuantizer
from .dataset import AudioDataset


def create_speech_encoder(cfg, device):
    """Loads E0 (SpeechEncoder) with HuBERT + KMeans."""
    logging.info(f"Loading continuous encoder and KMeans for '{cfg.model.name}'...")
    # SpeechEncoder from the library loads the dense model and the quantizer
    return SpeechEncoder.from_textlesslib(
        name=cfg.model.name,
        layer=cfg.model.layer,
        vocab_size=cfg.model.vocab_size,
        deduplicate=True,
        kind_kmeans=cfg.model.kind_kmeans
    ).to(device)


def train_quantizer(cfg):
    """
    Trains the augmentation-invariant discrete representation (E1) using CTC loss
    against pseudo-labels generated from unmodified audio using E0 (KMeans).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Pre-Trained KMeans Model (E0)
    E0 = create_speech_encoder(cfg, device)
    E0.eval() # E0 is always frozen
    
    # We can decouple the upstream model from E0 to get hidden states for E1
    upstream_encoder = E0.dense
    upstream_encoder.eval() # Upstream is always frozen
    
    # Initialize our learnable invariant quantizer (E1)
    vocab_size = cfg.model.vocab_size
    # +1 for the CTC blank token
    E1 = RobustQuantizer(
        input_dim=768, # HuBERT Base
        hidden_dim=cfg.model.quantizer.hidden_dim, 
        num_codes=vocab_size + 1 
    ).to(device)
    
    # Dataset and DataLoader
    # Clean dataset (no augmentations) for E0
    clean_dataset = AudioDataset(
        root=cfg.dataset.root, 
        split=cfg.dataset.train_split, 
        augment=False
    )
    
    # Augmentation pipeline is defined in AudioDataset if augment=True
    noise_dir = cfg.dataset.get("noise_dir", None)
    aug_dataset = AudioDataset(
        root=cfg.dataset.root, 
        split=cfg.dataset.train_split, 
        augment=True,
        config=cfg.dataset.augmentations,
        noise_dir=noise_dir
    )
    
    def collate_fn_paired(batch):
        # batch is list of (clean_waveform, augmented_waveform)
        clean_waves = [b[0] for b in batch]
        aug_waves = [b[1] for b in batch]
        
        max_clean_len = max(w.shape[-1] for w in clean_waves)
        max_aug_len = max(w.shape[-1] for w in aug_waves)
        
        padded_clean = torch.zeros(len(batch), 1, max_clean_len)
        padded_aug = torch.zeros(len(batch), 1, max_aug_len)
        
        clean_lens = torch.zeros(len(batch), dtype=torch.long)
        aug_lens = torch.zeros(len(batch), dtype=torch.long)
        
        for i, (cw, aw) in enumerate(zip(clean_waves, aug_waves)):
            clean_lens[i] = cw.shape[-1]
            padded_clean[i, :, :cw.shape[-1]] = cw
            aug_lens[i] = aw.shape[-1]
            padded_aug[i, :, :aw.shape[-1]] = aw
            
        # Squeeze the channel dim if it's 1 since HuBERT expects [batch, time]
        padded_clean = padded_clean.squeeze(1)
        padded_aug = padded_aug.squeeze(1)
            
        return padded_clean, clean_lens, padded_aug, aug_lens

    dataloader = DataLoader(
        aug_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_paired,
        num_workers=cfg.dataset.num_workers
    )

    optimizer = optim.Adam(E1.parameters(), lr=cfg.training.learning_rate)
    
    # CTC loss expects (T, N, C) for inputs
    ctc_loss = nn.CTCLoss(blank=vocab_size, zero_infinity=True)

    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    
    tensorboard_dir = os.path.join(cfg.training.checkpoint_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)

    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    resume_path = cfg.training.get("resume_from", None)
    if resume_path is not None:
        logging.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        E1.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        logging.info(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")

    scheduler = None
    if cfg.training.lr_scheduler is not None:
        scheduler = hydra.utils.instantiate(cfg.training.lr_scheduler, optimizer=optimizer)
    epoch = start_epoch
    for epoch in range(start_epoch, cfg.training.epochs+start_epoch):
        E1.train()
        total_loss = 0.0
        
        for batch_idx, (clean_audio, clean_lens, aug_audio, aug_lens) in enumerate(dataloader):
            clean_audio = clean_audio.to(device)
            clean_lens = clean_lens.to(device)
            aug_audio = aug_audio.to(device)
            aug_lens = aug_lens.to(device)

            # Log audio samples at the first batch of each epoch
            if batch_idx == 0 and epoch % 10 == 0:
                i = random.randint(0, clean_audio.shape[0] - 1)
                c_len = clean_lens[i].item()
                a_len = aug_lens[i].item()
                clean_sample = clean_audio[i, :c_len].unsqueeze(0).cpu()
                aug_sample = aug_audio[i, :a_len].unsqueeze(0).cpu()
                writer.add_audio(f"Audio/clean_sample_{i}", clean_sample, epoch, sample_rate=16000)
                writer.add_audio(f"Audio/augmented_sample_{i}", aug_sample, epoch, sample_rate=16000)

            # Target Generation using E0 (Clean Audio)
            with torch.no_grad():
                clean_outputs = E0(clean_audio, lengths=clean_lens, formatted=True)
                
                target_sequences = []
                target_lengths = []
                for out in clean_outputs:
                    units = torch.tensor(out['units'], dtype=torch.long, device=device)
                    target_sequences.append(units)
                    target_lengths.append(len(units))
                    
                # Concatenate targets into a single 1D tensor as required by CTCLoss
                flat_targets = torch.cat(target_sequences)
                target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)

            # Prediction using E1 (Augmented Audio)
            optimizer.zero_grad()
            
            with torch.no_grad():
                # Get the un-quantized representations from HuBERT
                aug_feats, out_aug_lens = upstream_encoder(aug_audio, lengths=aug_lens)
                aug_feats = aug_feats.clone()
                
            # Forward through E1 (our MLP)
            logits = E1(aug_feats) # [batch, seq_len, num_codes]
            
            # log_softmax is needed for CTC
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Permute to [seq_len, batch, num_codes] inside for CTC
            log_probs = log_probs.permute(1, 0, 2)
            
            # Input lengths are the number of frames generated by HuBERT
            # If out_aug_lens is None (all valid), use the full seq_len
            if out_aug_lens is None:
                out_aug_lens = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long, device=device)
            else:
                out_aug_lens = out_aug_lens.to(device)

            # CTC Loss
            loss = ctc_loss(log_probs, flat_targets, out_aug_lens, target_lengths)
            
            loss.backward()
            optimizer.step()
            if cfg.training.lr_scheduler is not None and epoch >= cfg.training.lr_scheduler_start_epoch:
                scheduler.step()
            
            if batch_idx % cfg.training.log_interval == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("Loss/CTC_Batch", loss.item(), global_step)
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("LR/learning_rate", current_lr, global_step)
                logging.info(f"Epoch {epoch} | Batch {batch_idx} | CTC Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/CTC_Epoch", avg_loss, epoch)
        logging.info(f"--- Epoch {epoch} Complete | Avg CTC Loss: {avg_loss:.4f} ---")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': E1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(cfg.training.checkpoint_dir, "E1_best.pt"))
            logging.info(f"    New best model saved (loss={best_loss:.4f})")

    # Save last model
    logging.info("Training complete.")
    torch.save({
        'epoch': epoch,
        'model_state_dict': E1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }, os.path.join(cfg.training.checkpoint_dir, "E1_last.pt"))
    writer.close()
    
    # ITERATIVE REFINEMENT TODO
