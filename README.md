# RoundingErrorEstimation

## 🧠 Training Script: `train_network.sh`

This Bash script is used to launch the training of a neural network on the MNIST dataset using PyTorch.

### 📄 Description

The script:

- Configures training parameters (epochs, batch size, checkpoint directory, etc.).
- Sets the Python import path to ensure proper module resolution (`PYTHONPATH`).
- Runs the training script located at `propagate_intervals/train.py`.

### ▶️ Usage

```bash
bash train_network.sh


* `propagate_intervals` - the code for experiment from the Section 3
* `lin_opt` - the code for linear optimisation experiments, Section 6 (Experiments) 
