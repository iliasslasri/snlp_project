import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import copy
import logging
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'speech_encoder', 'src'))

from speech_encoder import SpeechEncoder
from .models import RobustQuantizer
from .dataset import AudioDataset
from .utils import unit_edit_distance


def collate_fn_paired(batch):
    """Collate (clean, augmented) pairs into padded batch tensors.
    Must be defined at module level (not inside a function) for Windows
    multiprocessing compatibility (spawn requires pickling).
    """
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

    # HuBERT expects [batch, time]
    padded_clean = padded_clean.squeeze(1)
    padded_aug = padded_aug.squeeze(1)

    return padded_clean, clean_lens, padded_aug, aug_lens


def create_speech_encoder(cfg, device):
    """Loads E0 (SpeechEncoder) with HuBERT + KMeans."""
    logging.info(f"Loading continuous encoder and KMeans for '{cfg.model.name}'...")
    # SpeechEncoder from the library loads the dense model and the quantizer
    print("Device : ", device)
    return SpeechEncoder.from_textlesslib(
        name=cfg.model.name,
        layer=cfg.model.layer,
        vocab_size=cfg.model.vocab_size,
        deduplicate=True,
        kind_kmeans=cfg.model.kind_kmeans
    ).to(device)


@torch.no_grad()
def _evaluate_ued(E_teacher, E_student, upstream_encoder, dataloader, device):
    """Compute UED between E_teacher on clean and E_student on augmented audio."""
    E_teacher.eval()
    E_student.eval()
    upstream_encoder.eval()

    orig_units_all = []
    aug_units_all  = []

    for clean_audio, clean_lens, aug_audio, aug_lens in tqdm(dataloader, desc="  UED eval", leave=False):
        clean_audio = clean_audio.to(device)
        clean_lens  = clean_lens.to(device)
        aug_audio   = aug_audio.to(device)
        aug_lens    = aug_lens.to(device)

        clean_outputs = E_teacher(clean_audio, lengths=clean_lens, formatted=True)
        for out in clean_outputs:
            orig_units_all.append(out['units'])

        aug_feats, _ = upstream_encoder(aug_audio, lengths=aug_lens)
        logits = E_student(aug_feats)
        preds  = logits.argmax(dim=-1)
        for i in range(preds.shape[0]):
            aug_units_all.append(preds[i].cpu().tolist())

    return unit_edit_distance(orig_units_all, aug_units_all)


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
    aug_dataset = AudioDataset(
        root=cfg.dataset.root, 
        split=cfg.dataset.train_split, 
        augment=True,
        config=cfg.dataset.augmentations
    )

    dataloader = DataLoader(
        aug_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_paired,
        num_workers=0  # 0 required on Windows (spawn multiprocessing incompatibility)
    )

    optimizer = optim.Adam(E1.parameters(), lr=cfg.training.learning_rate)
    
    # CTC loss expects (T, N, C) for inputs
    ctc_loss = nn.CTCLoss(blank=vocab_size, zero_infinity=True)

    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    
    # Enable Tensorboard logging
    tensorboard_dir = os.path.join(cfg.training.checkpoint_dir, "runs")
    writer = SummaryWriter(log_dir=tensorboard_dir)

    logging.info("Starting training of Invariant Quantizer...")

    def _run_training_loop(E_teacher, E_student, optimizer, label):
        """Run cfg.training.epochs of CTC training. E_teacher is always frozen."""
        epoch_pbar = tqdm(range(cfg.training.epochs), desc=f"[{label}] Epochs", unit="epoch")

        for epoch in epoch_pbar:
            E_student.train()
            total_loss = 0.0

            batch_pbar = tqdm(dataloader, desc=f"  Epoch {epoch}", unit="batch", leave=False)

            for batch_idx, (clean_audio, clean_lens, aug_audio, aug_lens) in enumerate(batch_pbar):
                clean_audio = clean_audio.to(device)
                clean_lens = clean_lens.to(device)
                aug_audio = aug_audio.to(device)
                aug_lens = aug_lens.to(device)

                # Target Generation using E_teacher (Clean Audio)
                with torch.no_grad():
                    clean_outputs = E_teacher(clean_audio, lengths=clean_lens, formatted=True)

                    target_sequences = []
                    target_lengths = []
                    for out in clean_outputs:
                        units = torch.tensor(out['units'], dtype=torch.long, device=device)
                        target_sequences.append(units)
                        target_lengths.append(len(units))

                    flat_targets = torch.cat(target_sequences)
                    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)

                optimizer.zero_grad()

                with torch.no_grad():
                    aug_feats, out_aug_lens = upstream_encoder(aug_audio, lengths=aug_lens)
                    aug_feats = aug_feats.clone()

                logits = E_student(aug_feats)
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.permute(1, 0, 2)

                if out_aug_lens is None:
                    out_aug_lens = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long, device=device)
                else:
                    out_aug_lens = out_aug_lens.to(device)

                loss = ctc_loss(log_probs, flat_targets, out_aug_lens, target_lengths)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update inner bar with current loss
                batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

                if batch_idx % cfg.training.log_interval == 0:
                    global_step = epoch * len(dataloader) + batch_idx
                    writer.add_scalar(f"Loss/{label}_CTC_Batch", loss.item(), global_step)
                    logging.info(f"[{label}] Epoch {epoch} | Batch {batch_idx} | CTC Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(dataloader)
            writer.add_scalar(f"Loss/{label}_CTC_Epoch", avg_loss, epoch)
            logging.info(f"--- [{label}] Epoch {epoch} Complete | Avg CTC Loss: {avg_loss:.4f} ---")

            # Update outer bar with avg loss
            epoch_pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

            torch.save(E_student.state_dict(), os.path.join(cfg.training.checkpoint_dir, f"{label}_epoch_{epoch}.pt"))

    # --- Step 1: Non-iterative (Section 4.1) ---
    _run_training_loop(E_teacher=E0, E_student=E1, optimizer=optimizer, label="E1")

    logging.info("Training complete.")
    torch.save(E1.state_dict(), os.path.join(cfg.training.checkpoint_dir, "E1_best.pt"))

    # --- Step 2: Iterative refinement (Section 4.2) ---
    # Upon E1 convergence, freeze it and use it as the new teacher to train E2,
    # then E2 teaches E3, etc. Only replace after full convergence.
    num_iterations = getattr(cfg.training, 'num_iterations', 1)

    E_prev = E1
    for iteration in range(2, num_iterations + 1):
        logging.info(f"=== Iterative refinement: training E{iteration} ===")

        # Freeze the converged teacher
        E_teacher = copy.deepcopy(E_prev)
        E_teacher.eval()
        for p in E_teacher.parameters():
            p.requires_grad_(False)

        # Fresh student for this iteration
        E_next = RobustQuantizer(
            input_dim=768,
            hidden_dim=cfg.model.quantizer.hidden_dim,
            num_codes=vocab_size + 1
        ).to(device)
        optimizer_next = optim.Adam(E_next.parameters(), lr=cfg.training.learning_rate)

        _run_training_loop(E_teacher=E_teacher, E_student=E_next,
                           optimizer=optimizer_next, label=f"E{iteration}")

        # Evaluate UED after convergence
        ued = _evaluate_ued(E_teacher, E_next, upstream_encoder, dataloader, device)
        writer.add_scalar("UED/iterative", ued, iteration)
        logging.info(f"[E{iteration}] UED = {ued:.2f}")

        torch.save(E_next.state_dict(), os.path.join(cfg.training.checkpoint_dir, f"E{iteration}_best.pt"))
        E_prev = E_next  # converged student becomes teacher for next round

    writer.close()