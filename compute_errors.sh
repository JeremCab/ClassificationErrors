# # Path to config file
# CONFIG_PATH="configs/compute_errors.yaml"

# # Add project root to PYTHONPATH
# export PYTHONPATH=$(pwd)

# # Create output directory if it doesn't exist (parsed from YAML in the Python script)

# # Print configuration being used
# echo
# echo "Running error analysis with config: $CONFIG_PATH"

# # Run the script
# python optimization/compute_errors_lp.py --config $CONFIG_PATH



#!/bin/bash

# Path to config file
CONFIG_PATH="configs/compute_errors.yaml"

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd)

# Extract the optimization type from the YAML config
OPTIMIZATION=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['optimization'])")

# Choose the appropriate Python script
if [ "$OPTIMIZATION" == "linear" ]; then
    SCRIPT="compute_errors_lp.py"
elif [ "$OPTIMIZATION" == "non-linear" ]; then
    SCRIPT="compute_errors_nlp.py"
else
    echo "Unknown optimization type: $OPTIMIZATION"
    exit 1
fi

# Print what is going to be run
echo
echo "Running error analysis using: $SCRIPT"
echo "Config: $CONFIG_PATH"

# Run the appropriate script with the config
python optimization/$SCRIPT --config $CONFIG_PATH
