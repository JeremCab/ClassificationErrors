#!/bin/bash

# Configuration
EPOCHS=10
BATCH_SIZE=512
CHECKPOINT_DIR="checkpoints"

# Create the directory if it doesn't exist
mkdir -p $CHECKPOINT_DIR

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd)

# Print configuration
echo
echo "Training network with the following parameters:"
echo "  Epochs:           $EPOCHS"
echo "  Batch size:       $BATCH_SIZE"
echo "  Checkpoint dir:   $CHECKPOINT_DIR"
echo

# Run the training script using the correct path
python propagate_intervals/train.py \
  --batch_size $BATCH_SIZE \
  --num_epochs $EPOCHS \
  --checkpoint_dir $CHECKPOINT_DIR
