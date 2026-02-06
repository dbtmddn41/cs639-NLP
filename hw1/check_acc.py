#!/usr/bin/env python
import os

def compute_acc(output_file):
    with open(output_file) as f:
        preds = [int(line.strip()) for line in f]
    return preds

def read_labels(data_file):
    labels = []
    with open(data_file) as f:
        for line in f:
            tag = line.split(' ||| ')[0].strip()
            labels.append(int(tag))
    return labels

sst_dev_labels = read_labels('data/sst-dev.txt')
sst_test_labels = read_labels('data/sst-test.txt')
cfimdb_dev_labels = read_labels('data/cfimdb-dev.txt')
cfimdb_test_labels = read_labels('data/cfimdb-test.txt')

# Baseline
print('=== BASELINE ===')
sst_dev = compute_acc('baseline/sst-dev-output.txt')
sst_test = compute_acc('baseline/sst-test-output.txt')
cfimdb_dev = compute_acc('baseline/cfimdb-dev-output.txt')

print(f'SST Dev: {sum(p == l for p, l in zip(sst_dev, sst_dev_labels)) / len(sst_dev_labels):.4f}')
print(f'SST Test: {sum(p == l for p, l in zip(sst_test, sst_test_labels)) / len(sst_test_labels):.4f}')
print(f'CF-IMDB Dev: {sum(p == l for p, l in zip(cfimdb_dev, cfimdb_dev_labels)) / len(cfimdb_dev_labels):.4f}')

# 9089214606
print()
print('=== 9089214606 (95) ===')
sst_dev = compute_acc('9089214606/sst-dev-output.txt')
sst_test = compute_acc('9089214606/sst-test-output.txt')
cfimdb_dev = compute_acc('9089214606/cfimdb-dev-output.txt')

print(f'SST Dev: {sum(p == l for p, l in zip(sst_dev, sst_dev_labels)) / len(sst_dev_labels):.4f}')
print(f'SST Test: {sum(p == l for p, l in zip(sst_test, sst_test_labels)) / len(sst_test_labels):.4f}')
print(f'CF-IMDB Dev: {sum(p == l for p, l in zip(cfimdb_dev, cfimdb_dev_labels)) / len(cfimdb_dev_labels):.4f}')

# 100 (BiLSTM + Attention)
import os
if os.path.exists('100/sst-dev-output.txt'):
    print()
    print('=== 100 (BiLSTM + Attention) ===')
    sst_dev = compute_acc('100/sst-dev-output.txt')
    sst_test = compute_acc('100/sst-test-output.txt')
    cfimdb_dev = compute_acc('100/cfimdb-dev-output.txt')

    sst_dev_acc_100 = sum(p == l for p, l in zip(sst_dev, sst_dev_labels)) / len(sst_dev_labels)
    sst_test_acc_100 = sum(p == l for p, l in zip(sst_test, sst_test_labels)) / len(sst_test_labels)
    cfimdb_dev_acc_100 = sum(p == l for p, l in zip(cfimdb_dev, cfimdb_dev_labels)) / len(cfimdb_dev_labels)
    
    print(f'SST Dev: {sst_dev_acc_100:.4f}')
    print(f'SST Test: {sst_test_acc_100:.4f}')
    print(f'CF-IMDB Dev: {cfimdb_dev_acc_100:.4f}')

# 101 (BiLSTM only)
if os.path.exists('101/sst-dev-output.txt'):
    print()
    print('=== 101 (BiLSTM only) ===')
    sst_dev = compute_acc('101/sst-dev-output.txt')
    sst_test = compute_acc('101/sst-test-output.txt')
    cfimdb_dev = compute_acc('101/cfimdb-dev-output.txt')

    sst_dev_acc_101 = sum(p == l for p, l in zip(sst_dev, sst_dev_labels)) / len(sst_dev_labels)
    sst_test_acc_101 = sum(p == l for p, l in zip(sst_test, sst_test_labels)) / len(sst_test_labels)
    cfimdb_dev_acc_101 = sum(p == l for p, l in zip(cfimdb_dev, cfimdb_dev_labels)) / len(cfimdb_dev_labels)
    
    print(f'SST Dev: {sst_dev_acc_101:.4f}')
    print(f'SST Test: {sst_test_acc_101:.4f}')
    print(f'CF-IMDB Dev: {cfimdb_dev_acc_101:.4f}')

# 102 (DAN + Attention)
if os.path.exists('102/sst-dev-output.txt'):
    print()
    print('=== 102 (DAN + Attention) ===')
    sst_dev = compute_acc('102/sst-dev-output.txt')
    sst_test = compute_acc('102/sst-test-output.txt')
    cfimdb_dev = compute_acc('102/cfimdb-dev-output.txt')

    sst_dev_acc_102 = sum(p == l for p, l in zip(sst_dev, sst_dev_labels)) / len(sst_dev_labels)
    sst_test_acc_102 = sum(p == l for p, l in zip(sst_test, sst_test_labels)) / len(sst_test_labels)
    cfimdb_dev_acc_102 = sum(p == l for p, l in zip(cfimdb_dev, cfimdb_dev_labels)) / len(cfimdb_dev_labels)
    
    print(f'SST Dev: {sst_dev_acc_102:.4f}')
    print(f'SST Test: {sst_test_acc_102:.4f}')
    print(f'CF-IMDB Dev: {cfimdb_dev_acc_102:.4f}')

print()
print('=== Summary (Improvement over Baseline 39.87% / 91.84%) ===')
print(f'100 (BiLSTM+Attn): SST +{(sst_dev_acc_100 - 0.3987)*100:.2f}%, IMDB +{(cfimdb_dev_acc_100 - 0.9184)*100:.2f}%')
print(f'101 (BiLSTM only): SST +{(sst_dev_acc_101 - 0.3987)*100:.2f}%, IMDB +{(cfimdb_dev_acc_101 - 0.9184)*100:.2f}%')
print(f'102 (DAN+Attn):    SST +{(sst_dev_acc_102 - 0.3987)*100:.2f}%, IMDB +{(cfimdb_dev_acc_102 - 0.9184)*100:.2f}%')
