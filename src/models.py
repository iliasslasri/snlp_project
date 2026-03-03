import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel


class RobustQuantizer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_codes=500):
        super().__init__()
        # 3 fully connected layers with LeakyReLU per Appendix B of the paper
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, num_codes)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        logits = self.encoder(x)
        return logits
