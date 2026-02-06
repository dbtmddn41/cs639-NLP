#!/bin/bash
# Step 0. Change this to your campus ID
CAMPUSID='9089214606'
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
OUT_DIR="${SCRIPT_DIR}"

# Activate virtual environment if it exists
if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
    source "${ROOT_DIR}/.venv/bin/activate"
fi

# Step 1. Download GloVe if not present
if [ ! -f "${ROOT_DIR}/glove.6B.300d.txt" ]; then
    if command -v wget >/dev/null 2>&1; then
        wget -O "${ROOT_DIR}/glove.6B.zip" https://nlp.stanford.edu/data/glove.6B.zip
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "${ROOT_DIR}/glove.6B.zip" https://nlp.stanford.edu/data/glove.6B.zip
    else
        echo "Error: neither wget nor curl is available." >&2
        exit 1
    fi
    unzip -o "${ROOT_DIR}/glove.6B.zip" -d "${ROOT_DIR}"
fi

# Step 2. Train models on two datasets.
##  2.0. Run baseline (default settings)
echo "========================================"
echo "Training BASELINE (default settings)..."
echo "========================================"
BASELINE_DIR="${ROOT_DIR}/baseline"
mkdir -p "${BASELINE_DIR}"

PREF='sst'
python "${SCRIPT_DIR}/main.py" \
    --train "${ROOT_DIR}/data/${PREF}-train.txt" \
    --dev "${ROOT_DIR}/data/${PREF}-dev.txt" \
    --test "${ROOT_DIR}/data/${PREF}-test.txt" \
    --dev_output "${BASELINE_DIR}/${PREF}-dev-output.txt" \
    --test_output "${BASELINE_DIR}/${PREF}-test-output.txt" \
    --model "${BASELINE_DIR}/${PREF}-model.pt"

PREF='cfimdb'
python "${SCRIPT_DIR}/main.py" \
    --train "${ROOT_DIR}/data/${PREF}-train.txt" \
    --dev "${ROOT_DIR}/data/${PREF}-dev.txt" \
    --test "${ROOT_DIR}/data/${PREF}-test.txt" \
    --dev_output "${BASELINE_DIR}/${PREF}-dev-output.txt" \
    --test_output "${BASELINE_DIR}/${PREF}-test-output.txt" \
    --model "${BASELINE_DIR}/${PREF}-model.pt"

##  2.1. Run experiments on SST
echo "========================================"
echo "Training on SST dataset..."
echo "========================================"
PREF='sst'
python "${SCRIPT_DIR}/main.py" \
    --train "${ROOT_DIR}/data/${PREF}-train.txt" \
    --dev "${ROOT_DIR}/data/${PREF}-dev.txt" \
    --test "${ROOT_DIR}/data/${PREF}-test.txt" \
    --emb_file "${ROOT_DIR}/glove.6B.300d.txt" \
    --emb_size 300 \
    --use_lstm \
    --use_attention \
    --lstm_hidden 150 \
    --lstm_layers 1 \
    --hid_size 256 \
    --hid_layer 2 \
    --attention_heads 1 \
    --pooling_method "attention" \
    --pooling_norm \
    --ff_layernorm \
    --residual \
    --word_drop 0.15 \
    --emb_drop 0.4 \
    --hid_drop 0.5 \
    --label_smoothing 0.1 \
    --optimizer "adamw" \
    --weight_decay 1e-3 \
    --scheduler "plateau" \
    --lr_gamma 0.5 \
    --min_lr 1e-6 \
    --grad_clip 1.0 \
    --early_stop_patience 6 \
    --max_train_epoch 50 \
    --batch_size 32 \
    --lrate 0.001 \
    --eval_niter 200 \
    --dev_output "${OUT_DIR}/${PREF}-dev-output.txt" \
    --test_output "${OUT_DIR}/${PREF}-test-output.txt" \
    --model "${OUT_DIR}/${PREF}-model.pt"

##  2.2 Run experiments on CF-IMDB
echo "========================================"
echo "Training on CF-IMDB dataset..."
echo "========================================"
PREF='cfimdb'
python "${SCRIPT_DIR}/main.py" \
    --train "${ROOT_DIR}/data/${PREF}-train.txt" \
    --dev "${ROOT_DIR}/data/${PREF}-dev.txt" \
    --test "${ROOT_DIR}/data/${PREF}-test.txt" \
    --emb_file "${ROOT_DIR}/glove.6B.300d.txt" \
    --emb_size 300 \
    --use_lstm \
    --use_attention \
    --lstm_hidden 128 \
    --lstm_layers 1 \
    --hid_size 256 \
    --hid_layer 2 \
    --attention_heads 1 \
    --pooling_method "attention" \
    --pooling_norm \
    --ff_layernorm \
    --residual \
    --word_drop 0.1 \
    --emb_drop 0.3 \
    --hid_drop 0.4 \
    --label_smoothing 0.05 \
    --optimizer "adamw" \
    --weight_decay 1e-4 \
    --scheduler "plateau" \
    --lr_gamma 0.5 \
    --min_lr 1e-6 \
    --grad_clip 1.0 \
    --early_stop_patience 5 \
    --max_train_epoch 20 \
    --batch_size 32 \
    --lrate 0.001 \
    --eval_niter 200 \
    --dev_output "${OUT_DIR}/${PREF}-dev-output.txt" \
    --test_output "${OUT_DIR}/${PREF}-test-output.txt" \
    --model "${OUT_DIR}/${PREF}-model.pt"


# Step 3. Prepare submission
for file in 'main.py' 'model.py' 'vocab.py' 'setup.py' 'run_exp.sh'; do
    if [ -f "${SCRIPT_DIR}/$file" ]; then
        cp "${SCRIPT_DIR}/$file" "${OUT_DIR}/"
    fi
done

python "${ROOT_DIR}/prepare_submit.py" "${OUT_DIR}" ${CAMPUSID}

echo "========================================"
echo "Done!"
echo "========================================"

