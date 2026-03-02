import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel

class UpstreamEncoder(nn.Module):
    def __init__(self, model_name="facebook/hubert-base-ls960", layer=9, frozen=True):
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        
        if "hubert" in model_name:
            self.model = HubertModel.from_pretrained(model_name)
        elif "wav2vec" in model_name:
            self.model = Wav2Vec2Model.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: [batch, time]
        outputs = self.model(x, output_hidden_states=True)
        # Extract features from the specified layer
        # hidden_states is a tuple, index 0 is embedding, layers start at 1
        features = outputs.hidden_states[self.layer] 
        return features

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
