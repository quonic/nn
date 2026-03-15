package xor_example

import nn "../../"
import "core:fmt"
import "core:math/rand"
import "core:os"

print_training_progress :: proc(progress: nn.Training_Progress) {
	metrics := progress.metrics
	if metrics.accuracy_available {
		fmt.printf(
			"epoch=%d/%d loss=%0.6f mse=%0.6f mae=%0.6f accuracy=%0.2f%%\n",
			progress.epoch,
			progress.total_epochs,
			metrics.average_loss,
			metrics.mean_squared_error,
			metrics.mean_absolute_error,
			metrics.accuracy * 100,
		)
	} else {
		fmt.printf(
			"epoch=%d/%d loss=%0.6f mse=%0.6f mae=%0.6f\n",
			progress.epoch,
			progress.total_epochs,
			metrics.average_loss,
			metrics.mean_squared_error,
			metrics.mean_absolute_error,
		)
	}
}

brain_matches_example_setup :: proc(
	brain: ^nn.Brain,
	architecture: []int,
	config: nn.Brain_Config,
) -> bool {
	if len(brain.layers) != len(architecture) {
		return false
	}

	if brain.config.hidden_activation != config.hidden_activation ||
	   brain.config.output_activation != config.output_activation ||
	   brain.config.loss != config.loss {
		return false
	}

	for i := 0; i < len(architecture); i += 1 {
		if len(brain.layers[i].neurons) != architecture[i] {
			return false
		}
	}

	return true
}

main :: proc() {
	model_path := "xor-model.json"
	// We define a simple feedforward neural network architecture with 2 input neurons, two hidden layers of 4 neurons each, and 1 output neuron.
	architecture := [4]int{2, 4, 4, 1}
	// We configure the brain with ReLU activation for hidden layers, Sigmoid activation for the output layer, Binary Cross Entropy loss, and a learning rate of 0.15.
	config := nn.default_brain_config()
	config.hidden_activation = .ReLU
	config.output_activation = .Sigmoid
	config.loss = .Binary_Cross_Entropy

	brain: nn.Brain
	loaded := false
	if os.exists(model_path) {
		brain, loaded = nn.load_brain(model_path)
		if loaded && !brain_matches_example_setup(&brain, architecture[:], config) {
			nn.destroy_brain(&brain)
			brain = {}
			loaded = false
			fmt.printf(
				"Ignoring %s because it does not match the current example configuration\n",
				model_path,
			)
		}
	}

	if !loaded {
		// If we didn't load a model, we create a new brain.
		// We set a fixed random seed for reproducibility, so that the same model is trained each time.
		rand.reset_u64(1)
		brain = nn.make_brain_with_config(architecture[:], 0.15, config)
		fmt.println("Training XOR model...")
	}
	defer nn.destroy_brain(&brain)

	// raw_inputs are the 4 possible combinations of 2 binary inputs, and raw_labels are the expected outputs for the XOR function.
	raw_inputs := [4][2]f32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	// raw_labels are the expected outputs for the XOR function.
	raw_labels := [4][1]f32{{0}, {1}, {1}, {0}}

	// Prepare the training data and configuration.
	// If we loaded a model, we skip training and just evaluate the loaded model on the XOR inputs.
	inputs := make([][]f32, len(raw_inputs))
	labels := make([][]f32, len(raw_labels))
	training_config := nn.default_training_config()
	training_config.batch_size = 4
	training_config.shuffle_samples = true
	training_config.report_every = 1_000
	training_config.progress_callback = print_training_progress
	total_epochs := 20_000
	defer delete(inputs)
	defer delete(labels)

	// Convert raw_inputs and raw_labels to slices of slices for training.
	for i := 0; i < len(raw_inputs); i += 1 {
		inputs[i] = raw_inputs[i][:]
		labels[i] = raw_labels[i][:]
	}

	if !loaded {
		nn.train_with_config(&brain, inputs, labels, total_epochs, training_config)

		if nn.save_brain(&brain, model_path) {
			fmt.printf("Saved trained model to %s\n", model_path)
		} else {
			fmt.printf("Warning: failed to save trained model to %s\n", model_path)
		}
	} else {
		// If we loaded a model, we skip training and just evaluate the loaded model on the XOR inputs.
		fmt.printf("Loaded trained model from %s\n", model_path)
	}

	metrics := nn.compute_dataset_metrics(&brain, inputs, labels)
	if metrics.accuracy_available {
		fmt.printf(
			"final loss=%0.6f mse=%0.6f mae=%0.6f accuracy=%0.2f%%\n",
			metrics.average_loss,
			metrics.mean_squared_error,
			metrics.mean_absolute_error,
			metrics.accuracy * 100,
		)
	} else {
		fmt.printf(
			"final loss=%0.6f mse=%0.6f mae=%0.6f\n",
			metrics.average_loss,
			metrics.mean_squared_error,
			metrics.mean_absolute_error,
		)
	}

	fmt.println("XOR predictions after training:")
	for i := 0; i < len(raw_inputs); i += 1 {
		// Run the model on each of the 4 possible input combinations and print the output and predicted class.
		output := nn.run(&brain, raw_inputs[i][:])
		class := 0
		if output[0] >= 0.5 {
			class = 1
		}

		fmt.printf(
			"input=%v expected=%0.0f output=%0.4f class=%d\n",
			raw_inputs[i],
			raw_labels[i][0],
			output[0],
			class,
		)

		delete(output)
	}
}
