# A Neural Network Library for Odin

This is a small feedforward neural network library written in Odin. It supports configurable architectures, activations, and losses, as well as mini-batch training with progress callbacks. The library also includes functionality for saving and loading models in a versioned JSON format, along with dataset metrics and classification reports.

This is not optimized for performance or large-scale use. The focus is on clarity and correctness, with a simple API for training and inference.

This could be used for a small CPU ran model in a game for NPC's where a large library like TensorFlow or PyTorch would be overkill.

## Features

- Dense feedforward networks with configurable layer sizes
- Configurable activations: `ReLU`, `Leaky_ReLU`, `Sigmoid`, `Softmax`, `Tanh`, `Linear`
- Configurable losses: `Mean_Squared_Error`, `Mean_Absolute_Error`, `Binary_Cross_Entropy`, `Categorical_Cross_Entropy`
- Mini-batch training with optional sample shuffling
- Training progress callbacks with dataset metrics
- Model save/load through a versioned JSON persistence format
- Dataset metrics for loss, MSE, MAE, and classification accuracy
- Classification reports with confusion matrices for classification models
- Validation for unsupported activation/loss and architecture combinations

## Project Layout

```text
.
└── nn/
   ├── brain.odin
   ├── layer.odin
   ├── neuron.odin
   ├── persistence.odin
   ├── metrics_test.odin
   ├── persistence_test.odin
   └── examples/
       ├── xor/
       │   └── main.odin
       └── multiclass/
           └── main.odin
       └── tictactoe/
           ├── game.odin
           └── main.odin
```

## Requirements

- Odin installed and available on `PATH`

## Test

From the repository root:

```bash
odin test .
```

## Run The Examples

Run the XOR example:

```bash
odin run examples/xor
```

Output:

```text
Training XOR model...
epoch=1000/20000 loss=0.001266 mse=0.000002 mae=0.001265 accuracy=100.00%
epoch=2000/20000 loss=0.000472 mse=0.000000 mae=0.000472 accuracy=100.00%
epoch=3000/20000 loss=0.000276 mse=0.000000 mae=0.000276 accuracy=100.00%
epoch=4000/20000 loss=0.000191 mse=0.000000 mae=0.000191 accuracy=100.00%
epoch=5000/20000 loss=0.000144 mse=0.000000 mae=0.000144 accuracy=100.00%
epoch=6000/20000 loss=0.000115 mse=0.000000 mae=0.000115 accuracy=100.00%
epoch=7000/20000 loss=0.000095 mse=0.000000 mae=0.000095 accuracy=100.00%
epoch=8000/20000 loss=0.000081 mse=0.000000 mae=0.000080 accuracy=100.00%
epoch=9000/20000 loss=0.000070 mse=0.000000 mae=0.000070 accuracy=100.00%
epoch=10000/20000 loss=0.000061 mse=0.000000 mae=0.000061 accuracy=100.00%
epoch=11000/20000 loss=0.000055 mse=0.000000 mae=0.000055 accuracy=100.00%
epoch=12000/20000 loss=0.000049 mse=0.000000 mae=0.000049 accuracy=100.00%
epoch=13000/20000 loss=0.000045 mse=0.000000 mae=0.000045 accuracy=100.00%
epoch=14000/20000 loss=0.000041 mse=0.000000 mae=0.000041 accuracy=100.00%
epoch=15000/20000 loss=0.000038 mse=0.000000 mae=0.000038 accuracy=100.00%
epoch=16000/20000 loss=0.000035 mse=0.000000 mae=0.000035 accuracy=100.00%
epoch=17000/20000 loss=0.000033 mse=0.000000 mae=0.000032 accuracy=100.00%
epoch=18000/20000 loss=0.000030 mse=0.000000 mae=0.000030 accuracy=100.00%
epoch=19000/20000 loss=0.000028 mse=0.000000 mae=0.000028 accuracy=100.00%
epoch=20000/20000 loss=0.000027 mse=0.000000 mae=0.000027 accuracy=100.00%
Saved trained model to xor-model.json
final loss=0.000027 mse=0.000000 mae=0.000027 accuracy=100.00%
XOR predictions after training:
input=[0, 0] expected=0 output=0.0000 class=0
input=[0, 1] expected=1 output=1.0000 class=1
input=[1, 0] expected=1 output=1.0000 class=1
input=[1, 1] expected=0 output=0.0000 class=0
```

This example trains a binary XOR classifier, saves it to `xor-model.json`, and reuses the saved model on later runs when the saved architecture and config still match the example.

Run the multiclass example:

```bash
odin run examples/multiclass
```

Output:

```text
Training multiclass toy model...
epoch=500/4000 loss=0.00212 accuracy=100.00%
epoch=1000/4000 loss=0.00100 accuracy=100.00%
epoch=1500/4000 loss=0.00065 accuracy=100.00%
epoch=2000/4000 loss=0.00048 accuracy=100.00%
epoch=2500/4000 loss=0.00038 accuracy=100.00%
epoch=3000/4000 loss=0.00031 accuracy=100.00%
epoch=3500/4000 loss=0.00027 accuracy=100.00%
epoch=4000/4000 loss=0.00023 accuracy=100.00%
final loss=0.00023 mse=0.00000 mae=0.00046 accuracy=100.00%
confusion matrix (rows=expected, cols=predicted):
3 0 0
0 3 0
0 0 3
sample predictions:
input=[-1, -0.8] expected=0 predicted=0 output=[0.9997391, 0.0002128625, 4.805408e-05]
input=[-0.89999998, -1.1] expected=0 predicted=0 output=[0.99973458, 0.000241145768, 2.436322e-05]
input=[-1.2, -0.89999998] expected=0 predicted=0 output=[0.99982893, 0.000144844627, 2.623977e-05]
input=[1, -0.89999998] expected=1 predicted=1 output=[0.00013777254, 0.99979156, 7.070542e-05]
input=[0.8, -1.2] expected=1 predicted=1 output=[0.00028829311, 0.99967229, 3.9459806e-05]
input=[1.1, -0.69999999] expected=1 predicted=1 output=[0.00010557474, 0.9997409, 0.000153541798]
input=[0, 1] expected=2 predicted=2 output=[7.315735e-05, 8.0597209e-05, 0.9998462]
input=[-0.2, 1.2] expected=2 predicted=2 output=[7.763088e-05, 4.8715585e-05, 0.99987364]
input=[0.2, 0.8] expected=2 predicted=2 output=[8.0127196e-05, 0.00023255507, 0.9996873]
```

This example trains a 3-class toy classifier with `Softmax` and `Categorical_Cross_Entropy`, then prints a confusion matrix and sample predictions.

Run the tic-tac-toe example:

```bash
odin run examples/tictactoe
```

```text
Generated 627 perfect-play positions.
Training tic-tac-toe model...
epoch=100/400 loss=0.51604 accuracy=81.02%
epoch=200/400 loss=0.22329 accuracy=93.46%
epoch=300/400 loss=0.08370 accuracy=98.56%
epoch=400/400 loss=0.03985 accuracy=99.84%
Saved trained model to tictactoe-model.json
final loss=0.03985 mse=0.01051 mae=0.07171 accuracy=99.84%
confusion matrix (rows=expected, cols=predicted):
23 0 0 0 0 0 0 0 0
0 25 0 0 0 0 0 0 0
0 0 114 0 0 0 0 0 0
0 0 0 110 1 0 0 0 0
0 0 0 0 129 0 0 0 0
0 0 0 0 0 46 0 0 0
0 0 0 0 0 0 94 0 0
0 0 0 0 0 0 0 27 0
0 0 0 0 0 0 0 0 58

Play against the model.
Choose your side [X/O]: o
You are O. Squares use positions 1 through 9.

 1 | 2 | 3
---+---+---
 4 | 5 | 6
---+---+---
 7 | 8 | 9
Model plays X at square 1 (confidence=86.25%).

 X | 2 | 3
---+---+---
 4 | 5 | 6
---+---+---
 7 | 8 | 9
Choose a move [1-9]:
```

This example generates perfect-play tic-tac-toe positions with a minimax helper module, trains a 9-way move selector with `Softmax` and `Categorical_Cross_Entropy`, saves the model to `tictactoe-model.json`, and then lets you play against the trained network in the terminal.

## Minimal example

```odin
package main

import "nn"

main :: proc() {
 architecture := [3]int{2, 4, 1}
 config := nn.default_brain_config()
 config.hidden_activation = .ReLU
 config.output_activation = .Sigmoid
 config.loss = .Binary_Cross_Entropy

 brain := nn.make_brain_with_config(architecture[:], 0.15, config)
 defer nn.destroy_brain(&brain)

 inputs := [][]f32{
  {0, 0},
  {0, 1},
  {1, 0},
  {1, 1},
 }
 labels := [][]f32{
  {0},
  {1},
  {1},
  {0},
 }

 nn.train(&brain, inputs, labels, 10_000)
 output := nn.run(&brain, []f32{0, 1})
 defer delete(output)
}
```

## Main Public API

- `make_brain`
- `make_brain_with_config`
- `destroy_brain`
- `default_brain_config`
- `default_training_config`
- `validate_brain_config`
- `validate_brain_architecture`
- `brain_config_error`
- `brain_architecture_error`
- `train`
- `train_with_config`
- `run`
- `compute_cost`
- `compute_dataset_metrics`
- `compute_classification_report`
- `destroy_classification_report`
- `save_brain`
- `load_brain`
- `save_model_file`
- `load_model_file`

## Notes

- `Binary_Cross_Entropy` requires `Sigmoid` output activation.
- `Categorical_Cross_Entropy` requires `Softmax` output activation and at least two output neurons.
- Classification reports are available for `Sigmoid`, `Tanh`, and `Softmax` output models.
- Training asserts on invalid label vectors, including non-probability categorical targets.

## Current Status

- `odin test .` passes
- `odin run examples/xor` runs
- `odin run examples/multiclass` runs
- `odin run examples/tictactoe` should run
