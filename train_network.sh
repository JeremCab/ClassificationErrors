#!/bin/bash

# Configuration
EPOCHS=2
BATCH_SIZE=512
CHECKPOINT_DIR="checkpoints"
MODEL_NAME="mnist_dense_net"
# CHECKPOINT_PATH="${CHECKPOINT_DIR}/${MODEL_NAME}_epoch${EPOCHS}.pt"

# Create the directory if it doesn't exist
mkdir -p $CHECKPOINT_DIR

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd)

# Print configuration
echo "Training network with the following parameters:"
echo "  Epochs:           $EPOCHS"
echo "  Batch size:       $BATCH_SIZE"
echo "  Checkpoint dir:   $CHECKPOINT_DIR"
# echo "  Checkpoint path:  $CHECKPOINT_PATH"
echo

# Run training
python propagate_intervals/train.py \
  --batch_size $BATCH_SIZE \
  --num_epochs $EPOCHS \
  # --checkpoint_path $CHECKPOINT_PATH
