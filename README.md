# Classification Errors

Computing the classification errors between unquantized and quantized neural networks.

The experiment pipeline consists of the two following steps:

1. `train_network.sh` – for training the baseline model.
2. `compute_errors.sh` – for computing the errors between unquantized and quantized neural networks using linear and non-linear programming techniques.

---

## 1. `train_network.sh`

### 🎯 Purpose

This script trains a neural network on the MNIST dataset.

### ⚙️ Configuration Parameters

| Variable          | Description                               |
|-------------------|-------------------------------------------|
| `EPOCHS`          | Starting index of samples to evaluate     |
| `BATCH_SIZE`      | Ending index of samples to evaluate       |
| `CHECKPOINT_DIR`  | Number of quantization bits               |


### ▶️ Example Usage

```bash
./train_network.sh
```

---

## 2. `run_experiment.sh`

### 🎯 Purpose

This script performs a linear programming-based analysis to estimate worst-case quantization errors of a trained neural network.


### ⚙️ Configuration Parameters

| Variable      | Description                                                         |
|---------------|---------------------------------------------------------------------|

| `MODEL_NAME`  | Name of the trained model file (e.g., `mnist_smalldensenet_10.pt`)  |
| `START`       | Index of the first test sample to evaluate                          |
| `END`         | Index of the last test sample to evaluate                           |
| `BITS`        | Number of quantization bits                                         |
| `OUTPUT_DIR`  | Directory where results will be saved                               |


### ▶️ Example Usage

```bash
./compute_errors.sh
```