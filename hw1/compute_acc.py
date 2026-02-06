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

os.chdir('9089214606')

sst_dev = compute_acc('sst-dev-output.txt')
sst_test = compute_acc('sst-test-output.txt')
cfimdb_dev = compute_acc('cfimdb-dev-output.txt')
cfimdb_test = compute_acc('cfimdb-test-output.txt')

sst_dev_labels = read_labels('../data/sst-dev.txt')
sst_test_labels = read_labels('../data/sst-test.txt')
cfimdb_dev_labels = read_labels('../data/cfimdb-dev.txt')
cfimdb_test_labels = read_labels('../data/cfimdb-test.txt')

sst_dev_acc = sum(p == l for p, l in zip(sst_dev, sst_dev_labels)) / len(sst_dev_labels)
sst_test_acc = sum(p == l for p, l in zip(sst_test, sst_test_labels)) / len(sst_test_labels)
cfimdb_dev_acc = sum(p == l for p, l in zip(cfimdb_dev, cfimdb_dev_labels)) / len(cfimdb_dev_labels)
cfimdb_test_acc = sum(p == l for p, l in zip(cfimdb_test, cfimdb_test_labels)) / len(cfimdb_test_labels)

print(f'SST Dev: {sst_dev_acc:.4f}')
print(f'SST Test: {sst_test_acc:.4f}')
print(f'CF-IMDB Dev: {cfimdb_dev_acc:.4f}')
print(f'CF-IMDB Test: {cfimdb_test_acc:.4f}')
