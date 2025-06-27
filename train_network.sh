#!/bin/bash

# Path to config file
CONFIG_PATH="configs/train.yaml"

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd)

# Print configuration being used
echo
echo "Training network with config: $CONFIG_PATH"

# Run the training script with config path
python propagate_intervals/train.py --config $CONFIG_PATH
