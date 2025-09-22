#!/bin/bash

# Path to config file
CONFIG_PATH="configs/test.yaml"

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd)

# Print configuration being used
echo
echo "Testing network with config: $CONFIG_PATH"

# Run the training script with config path
python propagate_intervals/test.py --config $CONFIG_PATH
