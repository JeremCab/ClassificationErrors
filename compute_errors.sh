#!/bin/bash

# Configuration
MODEL_NAME="mnist_smalldensenet_10.pt"
START=0
END=3
BITS=16
OUTPUT_DIR="results"

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd)

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Print configuration
echo
echo "Running analysis with the following parameters:"
echo "  Start sample:    $START"
echo "  End sample:      $END"
echo "  Nb quant. bits:  $BITS"
echo "  Output dir:      $OUTPUT_DIR/"
echo "  Model name:      $MODEL_NAME"
echo

# Run the script
python optimization/compute_errors_lp.py \
  --model_name $MODEL_NAME\
  --start $START \
  --end $END \
  --bits $BITS \
  --outputdir $OUTPUT_DIR \
