import torch
import torch.nn as nn
import torch.nn.functional as F
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path, strict=True):
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        try:
            self.load_state_dict(ckpt['state_dict'], strict=strict)
        except RuntimeError as e:
            if strict:
                print(f'Warning: strict load failed ({e}). Retrying with strict=False.')
                self.load_state_dict(ckpt['state_dict'], strict=False)
            else:
                raise


def load_embedding(vocab, emb_file, emb_size):
    """Load pre-trained embeddings"""
    emb = np.random.uniform(-0.08, 0.08, (len(vocab), emb_size))
    
    if emb_file.endswith('.zip'):
        with zipfile.ZipFile(emb_file, 'r') as zf:
            file_name = zf.namelist()[0]
            with zf.open(file_name) as f:
                for line in f:
                    line = line.decode('utf-8').strip().split()
                    word = line[0]
                    if word in vocab:
                        vector = np.array([float(x) for x in line[1:]])
                        emb[vocab[word]] = vector
    else:
        with open(emb_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                word = line[0]
                if word in vocab:
                    vector = np.array([float(x) for x in line[1:]])
                    emb[vocab[word]] = vector
    
    return emb


class DanModel(BaseModel):
    """
    Simple BiLSTM model (no attention):
    - GloVe embeddings
    - Bidirectional LSTM
    - Avg+Max pooling
    - Feedforward layers with LayerNorm
    """
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """Define the model's parameters"""
        # Embedding layer
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size, padding_idx=self.vocab.pad_id)
        self.emb_drop = nn.Dropout(self.args.emb_drop)
        
        # BiLSTM layer
        lstm_hidden = getattr(self.args, 'lstm_hidden', self.args.hid_size // 2)
        num_layers = getattr(self.args, 'lstm_layers', 1)
        self.lstm = nn.LSTM(
            self.args.emb_size, 
            lstm_hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.args.hid_drop if num_layers > 1 else 0
        )
        self.lstm_drop = nn.Dropout(self.args.hid_drop)
        encoder_out_size = lstm_hidden * 2  # bidirectional
        
        # Pooling configuration: avgmax (avg + max concatenated)
        self.pooling_method = getattr(self.args, 'pooling_method', 'avgmax')
        if self.pooling_method == 'avgmax':
            pooled_size = encoder_out_size * 2
        else:
            pooled_size = encoder_out_size
        
        # Layer normalization after pooling
        self.pooling_norm = nn.LayerNorm(pooled_size) if getattr(self.args, 'pooling_norm', True) else None
        
        # Feedforward layers
        self.ff_layers = nn.ModuleList()
        self.ff_norms = nn.ModuleList() if getattr(self.args, 'ff_layernorm', True) else None
        self.use_residual = getattr(self.args, 'residual', True)
        
        input_size = pooled_size
        for i in range(self.args.hid_layer):
            self.ff_layers.append(nn.Linear(input_size, self.args.hid_size))
            if self.ff_norms is not None:
                self.ff_norms.append(nn.LayerNorm(self.args.hid_size))
            input_size = self.args.hid_size
        
        self.ff_act = nn.ReLU()
        self.ff_drop = nn.Dropout(self.args.hid_drop)
        
        # Output layer
        self.output = nn.Linear(input_size, self.tag_size)

    def init_model_parameters(self):
        """Initialize the model's parameters"""
        v = getattr(self.args, 'init_range', 0.08)
        for name, param in self.named_parameters():
            if 'embedding' in name:
                continue
            elif 'norm' in name:
                if 'weight' in name:
                    nn.init.ones_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            elif 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            elif param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -v, v)

    def copy_embedding_from_numpy(self):
        """Load pre-trained word embeddings"""
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb).float())
        self.embedding.weight.data[self.vocab.pad_id].zero_()

    def forward(self, x):
        """
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        batch_size, seq_len = x.size()
        
        # Create mask for padding tokens
        pad_mask = (x != self.vocab.pad_id).float()  # [batch_size, seq_length]
        
        # Get embeddings
        emb = self.embedding(x)  # [batch_size, seq_length, emb_size]
        emb = self.emb_drop(emb)
        
        # Word dropout during training
        if self.training and getattr(self.args, 'word_drop', 0) > 0:
            drop_mask = torch.bernoulli(
                torch.full((batch_size, seq_len), 1 - self.args.word_drop, device=x.device)
            ).float()
            word_mask = pad_mask * drop_mask
        else:
            word_mask = pad_mask
        
        # Apply word mask to embeddings
        emb_masked = emb * word_mask.unsqueeze(-1)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(emb_masked)  # [batch, seq_len, hidden*2]
        lstm_out = self.lstm_drop(lstm_out)
        encoder_out = lstm_out
        
        # Pooling
        mask_expanded = word_mask.unsqueeze(-1)
        if self.pooling_method == 'avgmax':
            # Average + Max pooling concatenation
            lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled_avg = (encoder_out * mask_expanded).sum(dim=1) / lengths
            encoder_masked = encoder_out * mask_expanded + (1 - mask_expanded) * (-1e9)
            pooled_max, _ = encoder_masked.max(dim=1)
            pooled = torch.cat([pooled_avg, pooled_max], dim=1)
        elif self.pooling_method == 'avg':
            lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = (encoder_out * mask_expanded).sum(dim=1) / lengths
        elif self.pooling_method == 'max':
            encoder_masked = encoder_out * mask_expanded + (1 - mask_expanded) * (-1e9)
            pooled, _ = encoder_masked.max(dim=1)
        else:  # sum
            pooled = (encoder_out * mask_expanded).sum(dim=1)
        
        if self.pooling_norm is not None:
            pooled = self.pooling_norm(pooled)
        
        # Feedforward layers
        hidden = pooled
        for i, layer in enumerate(self.ff_layers):
            out = layer(hidden)
            if self.ff_norms is not None:
                out = self.ff_norms[i](out)
            out = self.ff_act(out)
            out = self.ff_drop(out)
            if self.use_residual and out.shape == hidden.shape:
                hidden = hidden + out
            else:
                hidden = out
        
        # Output layer
        scores = self.output(hidden)
        
        return scores
