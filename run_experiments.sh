#!/usr/bin/env bash

CONFIG="configs/compute_errors.yaml"
VALUES=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for p in "${VALUES[@]}"; do
    echo "=== Running with p = $p ==="

    # Update p in YAML
    yq -yi ".p = $p" "$CONFIG"

    # (optional) confirm change
    echo "Updated p in YAML to: $(yq '.p' "$CONFIG")"

    # Run your script
    ./compute_errors.sh

    echo "=== Done p = $p ==="
done

