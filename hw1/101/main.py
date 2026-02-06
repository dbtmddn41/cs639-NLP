from collections import defaultdict
import os
import time
import random
import torch
import torch.nn as nn
import model as mn
import numpy as np
import argparse
from vocab import Vocab


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/sst-train.txt")
    parser.add_argument("--dev", type=str, default="data/sst-dev.txt")
    parser.add_argument("--test", type=str, default="data/sst-test.txt")
    parser.add_argument("--emb_file", type=str, default=None)
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--hid_size", type=int, default=256)
    parser.add_argument("--hid_layer", type=int, default=2)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--use_lstm", action="store_true", default=True)
    parser.add_argument("--no_lstm", action="store_false", dest="use_lstm")
    parser.add_argument("--use_attention", action="store_true", default=True)
    parser.add_argument("--no_attention", action="store_false", dest="use_attention")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument("--word_drop", type=float, default=0.1)
    parser.add_argument("--emb_drop", type=float, default=0.3)
    parser.add_argument("--hid_drop", type=float, default=0.4)
    parser.add_argument("--pooling_method", type=str, default="attention", 
                        choices=["sum", "avg", "max", "avgmax", "attention"])
    parser.add_argument("--pooling_norm", action="store_true", default=True)
    parser.add_argument("--ff_layernorm", action="store_true", default=True)
    parser.add_argument("--residual", action="store_true", default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lrate", type=float, default=0.001)
    parser.add_argument("--lrate_decay", type=float, default=0)
    parser.add_argument("--mrate", type=float, default=0.85)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adagrad", "adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau", "step", "cosine"])
    parser.add_argument("--lr_step_size", type=int, default=1)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--log_niter", type=int, default=100)
    parser.add_argument("--eval_niter", type=int, default=200)
    parser.add_argument("--model", type=str, default="model.pt")
    parser.add_argument("--dev_output", type=str, default="output.dev.txt")
    parser.add_argument("--test_output", type=str, default="output.test.txt")
    args = parser.parse_args()
    print(f"RUN: {vars(args)}")
    return args

def read_dataset(filename):
    dataset = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            dataset.append((words.split(' '), tag))
    return dataset

def convert_text_to_ids(dataset, word_vocab, tag_vocab):
    data = []
    for words, tag in dataset:
        word_ids = [word_vocab[w] for w in words]
        data.append((word_ids, tag_vocab[tag]))
    return data

def data_iter(data, batch_size, shuffle=True):
    """
    Randomly shuffle training data, and partition into batches.
    """
    if shuffle:
        np.random.shuffle(data)

    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tags = [data[i * batch_size + b][1] for b in range(cur_batch_size)]
        yield sents, tags

def pad_sentences(sents, pad_id):
    """
    Adding pad_id to sentences in a mini-batch.
    """
    max_len = max(len(s) for s in sents)
    aug_sents = [s + [pad_id] * (max_len - len(s)) for s in sents]
    return aug_sents

def compute_grad_norm(model, norm_type=2):
    """Compute the gradients' L2 norm"""
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        p_norm = p.grad.norm(norm_type) ** (norm_type)
        total_norm += p_norm
    return total_norm ** (1. / norm_type)


def compute_param_norm(model, norm_type=2):
    """Compute the model's parameters' L2 norm"""
    total_norm = 0.0
    for p in model.parameters():
        p_norm = p.norm(norm_type) ** (norm_type)
        total_norm += p_norm
    return total_norm ** (1. / norm_type)

def evaluate(dataset, model, device, tag_vocab=None, filename=None):
    """Evaluate test/dev set"""
    model.eval()
    predicts = []
    acc = 0
    with torch.no_grad():
        for words, tag in dataset:
            X = torch.LongTensor([words]).to(device)
            scores = model(X)
            y_pred = scores.argmax(1)[0].item()
            predicts.append(y_pred)
            acc += int(y_pred == tag)
    print(f'  -Accuracy: {acc/len(predicts):.4f} ({acc}/{len(predicts)})')
    if filename:
        with open(filename, 'w') as f:
            for y_pred in predicts:
                tag = tag_vocab.id2word[y_pred]
                f.write(f'{tag}\n')
        print(f'  -Save predictions to {filename}')
    model.train()
    return acc/len(predicts)

def main():
    args = get_args()
    _seed = os.environ.get("MINNN_SEED", 12341)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(_seed)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Read datasets
    train_text = read_dataset(args.train)
    dev_text = read_dataset(args.dev)
    test_text = read_dataset(args.test)
    
    # Build vocabularies
    word_vocab = Vocab(pad=True, unk=True)
    word_vocab.build(list(zip(*train_text))[0])
    tag_vocab = Vocab()
    tag_vocab.build(list(zip(*train_text))[1])
    
    # Convert text to ids
    train_data = convert_text_to_ids(train_text, word_vocab, tag_vocab)
    dev_data = convert_text_to_ids(dev_text, word_vocab, tag_vocab)
    test_data = convert_text_to_ids(test_text, word_vocab, tag_vocab)

    # Create model
    nwords = len(word_vocab)
    ntags = len(tag_vocab)
    print('nwords', nwords, 'ntags', ntags)
    model = mn.DanModel(args, word_vocab, len(tag_vocab)).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}, Trainable: {trainable_params:,}')
    
    # Loss function
    if args.label_smoothing > 0:
        loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        loss_func = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lrate, lr_decay=args.lrate_decay)

    # Scheduler
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_gamma, patience=2, min_lr=args.min_lr
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.scheduler == "cosine":
        total_steps = args.max_train_epoch * len(train_data) // args.batch_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.min_lr
        )

    # Training
    start_time = time.time()
    train_iter = 0
    train_loss = train_example = train_correct = 0
    best_records = (0, 0)  # [best_iter, best_accuracy]
    no_improve = 0
    stop_training = False
    
    for epoch in range(args.max_train_epoch):
        for batch in data_iter(train_data, batch_size=args.batch_size, shuffle=True):
            train_iter += 1

            X = pad_sentences(batch[0], word_vocab['<pad>'])
            X = torch.LongTensor(X).to(device)
            Y = torch.LongTensor(batch[1]).to(device)
            
            # Forward pass
            scores = model(X)
            loss = loss_func(scores, Y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            # Cosine scheduler updates per step
            if args.scheduler == "cosine" and scheduler is not None:
                scheduler.step()

            train_loss += loss.item() * len(batch[0])
            train_example += len(batch[0])
            Y_pred = scores.argmax(1)
            train_correct += (Y_pred == Y).sum().item()

            if train_iter % args.log_niter == 0:
                gnorm = compute_grad_norm(model)
                pnorm = compute_param_norm(model)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}, iter {train_iter}, train set: '
                    f'loss={train_loss/train_example:.4f}, '
                    f'accuracy={train_correct/train_example:.2f} ({train_correct}/{train_example}), '
                    f'gradient_norm={gnorm:.2f}, params_norm={pnorm:.2f}, '
                    f'lr={current_lr:.6f}, '
                    f'time={time.time()-start_time:.2f}s')
                train_loss = train_example = train_correct = 0

            if train_iter % args.eval_niter == 0:
                print(f'Evaluate dev data:')
                dev_accuracy = evaluate(dev_data, model, device)
                
                if dev_accuracy > best_records[1] + args.early_stop_min_delta:
                    print(f'  -Update best model at {train_iter}, dev accuracy={dev_accuracy:.4f}')
                    best_records = (train_iter, dev_accuracy)
                    model.save(args.model)
                    no_improve = 0
                else:
                    no_improve += 1

                if scheduler is not None and args.scheduler == "plateau":
                    scheduler.step(dev_accuracy)

                if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                    print(f'Early stopping at iter {train_iter} (no improvement for {no_improve} evals)')
                    stop_training = True
                    break

        if scheduler is not None and args.scheduler == "step":
            scheduler.step()

        if stop_training:
            break

    # Load best model and evaluate
    if not os.path.exists(args.model):
        model.save(args.model)
    model.load(args.model)
    print('Final evaluation on test set:')
    evaluate(test_data, model, device, tag_vocab, filename=args.test_output)
    print('Final evaluation on dev set:')
    evaluate(dev_data, model, device, tag_vocab, filename=args.dev_output)


if __name__ == '__main__':
    main()
