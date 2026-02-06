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
    """Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText)."""
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


class SelfAttention(nn.Module):
    """Self-attention layer for sequence pooling"""
    def __init__(self, hidden_size, attention_heads=1):
        super(SelfAttention, self).__init__()
        self.attention_heads = attention_heads
        self.attention = nn.Linear(hidden_size, attention_heads)

    def forward(self, hidden_states, mask=None):
        attn_scores = self.attention(hidden_states)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=1)

        if self.attention_heads == 1:
            pooled = (hidden_states * attn_weights).sum(dim=1)
        else:
            pooled = torch.einsum('bsh,bsd->bhd', attn_weights, hidden_states)
            pooled = pooled.view(pooled.size(0), -1)

        return pooled, attn_weights


class DanModel(BaseModel):
    """
    Deep Averaging Network with optional BiLSTM and self-attention pooling.
    """
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size, padding_idx=self.vocab.pad_id)
        self.emb_drop = nn.Dropout(self.args.emb_drop)

        self.use_lstm = getattr(self.args, 'use_lstm', False)
        self.use_attention = getattr(self.args, 'use_attention', False)

        if self.use_lstm:
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
            encoder_out_size = lstm_hidden * 2
        else:
            encoder_out_size = self.args.emb_size

        self.pooling_method = getattr(self.args, 'pooling_method', 'avg')
        if self.pooling_method == 'attention' and not self.use_attention:
            self.pooling_method = 'avg'

        if self.use_attention and self.pooling_method == 'attention':
            attention_heads = getattr(self.args, 'attention_heads', 1)
            self.attention = SelfAttention(encoder_out_size, attention_heads)
            pooled_size = encoder_out_size * attention_heads
        elif self.pooling_method == 'avgmax':
            pooled_size = encoder_out_size * 2
        else:
            pooled_size = encoder_out_size

        self.pooling_norm = nn.LayerNorm(pooled_size) if getattr(self.args, 'pooling_norm', False) else None

        self.ff_layers = nn.ModuleList()
        self.ff_norms = nn.ModuleList() if getattr(self.args, 'ff_layernorm', False) else None
        self.use_residual = getattr(self.args, 'residual', False)

        input_size = pooled_size
        for _ in range(self.args.hid_layer):
            self.ff_layers.append(nn.Linear(input_size, self.args.hid_size))
            if self.ff_norms is not None:
                self.ff_norms.append(nn.LayerNorm(self.args.hid_size))
            input_size = self.args.hid_size

        self.ff_act = nn.GELU()
        self.ff_drop = nn.Dropout(self.args.hid_drop)
        self.output = nn.Linear(input_size, self.tag_size)

    def init_model_parameters(self):
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
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb).float())
        self.embedding.weight.data[self.vocab.pad_id].zero_()

    def forward(self, x):
        batch_size, seq_len = x.size()

        pad_mask = (x != self.vocab.pad_id).float()

        emb = self.embedding(x)
        emb = self.emb_drop(emb)

        if self.training and getattr(self.args, 'word_drop', 0) > 0:
            drop_mask = torch.bernoulli(
                torch.full((batch_size, seq_len), 1 - self.args.word_drop, device=x.device)
            ).float()
            word_mask = pad_mask * drop_mask
        else:
            word_mask = pad_mask

        if self.use_lstm:
            emb_masked = emb * word_mask.unsqueeze(-1)
            lstm_out, _ = self.lstm(emb_masked)
            lstm_out = self.lstm_drop(lstm_out)
            encoder_out = lstm_out
        else:
            encoder_out = emb * word_mask.unsqueeze(-1)

        if self.use_attention and self.pooling_method == 'attention':
            pooled, _ = self.attention(encoder_out, word_mask)
        elif self.pooling_method == 'avgmax':
            mask_expanded = word_mask.unsqueeze(-1)
            lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled_avg = (encoder_out * mask_expanded).sum(dim=1) / lengths
            encoder_masked = encoder_out * mask_expanded + (1 - mask_expanded) * (-1e9)
            pooled_max, _ = encoder_masked.max(dim=1)
            pooled = torch.cat([pooled_avg, pooled_max], dim=1)
        elif self.pooling_method == 'avg':
            mask_expanded = word_mask.unsqueeze(-1)
            lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = (encoder_out * mask_expanded).sum(dim=1) / lengths
        elif self.pooling_method == 'max':
            mask_expanded = word_mask.unsqueeze(-1)
            encoder_masked = encoder_out * mask_expanded + (1 - mask_expanded) * (-1e9)
            pooled, _ = encoder_masked.max(dim=1)
        else:
            mask_expanded = word_mask.unsqueeze(-1)
            pooled = (encoder_out * mask_expanded).sum(dim=1)

        if self.pooling_norm is not None:
            pooled = self.pooling_norm(pooled)

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

        scores = self.output(hidden)

        return scores
