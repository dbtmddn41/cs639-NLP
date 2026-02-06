# Step 0. Output directory (baseline)
CAMPUSID='baseline'
OUTPUT_DIR="${CAMPUSID}"
mkdir -p "${OUTPUT_DIR}"

# Step 1. (Optional) Any preprocessing step, e.g., downloading pre-trained word embeddings
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# Step 2. Train models on two datasets.
##  2.1. Run experiments on SST
PREF='sst'
python main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_output "${OUTPUT_DIR}/${PREF}-dev-output.txt" \
    --test_output "${OUTPUT_DIR}/${PREF}-test-output.txt" \
    --model "${OUTPUT_DIR}/${PREF}-model.pt"

##  2.2 Run experiments on CF-IMDB
PREF='cfimdb'
python main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_output "${OUTPUT_DIR}/${PREF}-dev-output.txt" \
    --test_output "${OUTPUT_DIR}/${PREF}-test-output.txt" \
    --model "${OUTPUT_DIR}/${PREF}-model.pt"


# Step 3. Prepare submission (optional):
# Set PREPARE_SUBMIT=1 to enable packaging.
if [ "${PREPARE_SUBMIT:-0}" = "1" ]; then
    ##  3.1. Copy your code to the $OUTPUT_DIR folder
    for file in 'main.py' 'model.py' 'vocab.py' 'setup.py' 'run_exp.sh'; do
        if [ -f "$file" ]; then
            cp "$file" "${OUTPUT_DIR}/"
        fi
    done
    ##  3.2. Compress the $OUTPUT_DIR folder to $CAMPUSID.zip (containing only .py/.txt/.pdf/.sh files)
    python prepare_submit.py "${OUTPUT_DIR}" ${CAMPUSID}
    ##  3.3. Submit the zip file to Canvas (https://canvas.wisc.edu/courses/292771/assignments)! Congrats!
fi
