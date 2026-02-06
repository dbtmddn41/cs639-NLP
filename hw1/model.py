import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    # Initialize embedding matrix with random values
    emb = np.random.uniform(-0.08, 0.08, (len(vocab), emb_size))
    
    # Check if emb_file is a zip file
    if emb_file.endswith('.zip'):
        with zipfile.ZipFile(emb_file, 'r') as zf:
            # Get the first file in the zip
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
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        # Embedding layer
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size, padding_idx=self.vocab.pad_id)
        
        # Embedding dropout layer
        self.emb_drop = nn.Dropout(self.args.emb_drop)
        
        # Feedforward layers
        layers = []
        input_size = self.args.emb_size
        for i in range(self.args.hid_layer):
            layers.append(nn.Linear(input_size, self.args.hid_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.args.hid_drop))
            input_size = self.args.hid_size
        self.feedforward = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(self.args.hid_size, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        v = getattr(self.args, 'init_range', 0.08)
        for name, param in self.named_parameters():
            nn.init.uniform_(param, -v, v)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb).float())

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        # 1. Create mask for padding tokens
        pad_mask = (x != self.vocab.pad_id).float()  # [batch_size, seq_length]
        
        # 2. Get embeddings: [batch_size, seq_length, emb_size]
        emb = self.embedding(x)
        
        # Apply embedding dropout
        emb = self.emb_drop(emb)
        
        # 3. Word dropout: drop words by masking them out (not replacing with <unk>)
        if self.training and self.args.word_drop > 0:
            # Create dropout mask (1 = keep, 0 = drop)
            drop_mask = torch.bernoulli(torch.full((x.size(0), x.size(1)), 
                                                   1 - self.args.word_drop, 
                                                   device=x.device)).float()
            # Apply only to non-padding tokens
            final_mask = pad_mask * drop_mask
        else:
            final_mask = pad_mask
        
        # Expand dimensions for broadcasting: [batch_size, seq_length, 1]
        final_mask = final_mask.unsqueeze(-1)
        
        # 4. Pooling over sequence (with word dropout applied)
        if self.args.pooling_method == 'sum':
            pooled = (emb * final_mask).sum(dim=1)  # [batch_size, emb_size]
        elif self.args.pooling_method == 'avg':
            # Important: dropped words are excluded from both numerator and denominator
            lengths = final_mask.sum(dim=1).clamp(min=1e-9)  # [batch_size, 1]
            pooled = (emb * final_mask).sum(dim=1) / lengths  # [batch_size, emb_size]
        elif self.args.pooling_method == 'max':
            emb_masked = emb * final_mask + (1 - final_mask) * (-1e9)  # mask with large negative
            pooled, _ = emb_masked.max(dim=1)  # [batch_size, emb_size]
        
        # 5. Feedforward layers
        hidden = self.feedforward(pooled)
        
        # 6. Output layer
        scores = self.output(hidden)
        
        return scores
