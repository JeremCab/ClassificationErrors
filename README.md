# Classification Errors

Computing the classification errors between unquantized and quantized neural networks.

---

## `train_network.sh`

This script analyzes classification errors between an original neural network and its quantized counterpart using a direct evaluation strategy.


### 🎯 Purpose

To compute and compare real versus predicted classification errors by directly evaluating the original and quantized networks side-by-side.


### ⚙️ Configuration Parameters

| Variable      | Description                                                         |
|---------------|---------------------------------------------------------------------|
| `START`       | Starting index of samples to evaluate                               |
| `END`         | Ending index of samples to evaluate                                 |
| `BITS`        | Number of quantization bits                                         |
| `OUTPUT_DIR`  | Output directory for storing results                                |
| `MODEL_NAME`  | Name of the model used for evaluation                               |


### 🧠 What It Does

1. Sets `PYTHONPATH` to the current directory to ensure module resolution.
2. Creates the output directory if it doesn’t exist.
3. Prints a summary of the execution configuration.
4. Calls the Python script `lin_opt/compute_errors.py` with the given parameters:

```bash
python lin_opt/compute_errors.py \
  --start $START \
  --end $END \
  --bits $BITS \
  --outputdir $OUTPUT_DIR \
  $MODEL_NAME
```


### ▶️ Example Usage

```bash
./train_network.sh
```

Ensure the model name corresponds to a valid checkpoint, e.g., `mnist_dense_net` for `checkpoints/mnist_dense_net.pt`.


### 🛠 Tips

- Make the script executable:

```bash
chmod +x train_network.sh
```

- You can modify the parameters at the top of the script to fit your desired experiment setup.

---

## `run_experiment.sh`

This script performs a linear programming-based analysis to estimate worst-case quantization errors of a trained neural network.


### 🎯 Purpose

To validate the robustness of a quantized neural network by computing theoretical upper bounds on classification errors using linear programming.


### ⚙️ Configuration Parameters

| Variable      | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `MODEL_NAME`  | Name of the trained model file (e.g., `mnist_smalldensenet_10.pt`)         |
| `START`       | Index of the first test sample to evaluate                                 |
| `END`         | Index of the last test sample to evaluate                                  |
| `BITS`        | Number of quantization bits                                                |
| `OUTPUT_DIR`  | Directory where results will be saved                                      |


### 🧠 What It Does

1. Sets `PYTHONPATH` to the project root for proper module resolution.
2. Ensures the output directory exists.
3. Prints the current run configuration.
4. Executes the LP-based analysis script:

```bash
python optimization/compute_errors_lp.py \
  --model_name $MODEL_NAME \
  --start $START \
  --end $END \
  --bits $BITS \
  --outputdir $OUTPUT_DIR
```


### ▶️ Example Usage

```bash
./run_experiment.sh
```

Ensure the model file exists in the expected path, e.g., `checkpoints/mnist_smalldensenet_10.pt`.


### 🛠 Tips

- Make the script executable:

```bash
chmod +x run_experiment.sh
```

- You can edit the variables in the script to change the experiment's scope.
