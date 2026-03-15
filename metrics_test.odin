package nn

import "core:math"
import "core:testing"

@(private = "file")
callback_invocations: int

record_progress_callback :: proc(progress: Training_Progress) {
	_ = progress
	callback_invocations += 1
}

@(test)
test_validate_brain_config :: proc(t: ^testing.T) {
	config := default_brain_config()
	config.output_activation = .Sigmoid
	config.loss = .Binary_Cross_Entropy
	testing.expect(
		t,
		validate_brain_config(config),
		"sigmoid + binary cross entropy should be valid",
	)

	config.output_activation = .Linear
	testing.expect(
		t,
		!validate_brain_config(config),
		"linear + binary cross entropy should be invalid",
	)
	testing.expect_value(
		t,
		brain_config_error(config),
		"Binary_Cross_Entropy requires Sigmoid output activation",
	)

	config.output_activation = .Linear
	config.loss = .Mean_Absolute_Error
	testing.expect(
		t,
		validate_brain_config(config),
		"linear + mean absolute error should be valid",
	)
	config.hidden_activation = .Leaky_ReLU
	testing.expect(
		t,
		validate_brain_config(config),
		"leaky relu hidden activation should be valid",
	)

	softmax_config := default_brain_config()
	softmax_config.output_activation = .Softmax
	softmax_config.loss = .Categorical_Cross_Entropy
	valid_softmax_architecture := [2]int{4, 3}
	invalid_softmax_architecture := [2]int{4, 1}
	testing.expect(
		t,
		validate_brain_config(softmax_config),
		"softmax + categorical cross entropy should be valid",
	)
	testing.expect(
		t,
		validate_brain_architecture(valid_softmax_architecture[:], softmax_config),
		"softmax output with 3 classes should be valid",
	)
	testing.expect(
		t,
		!validate_brain_architecture(invalid_softmax_architecture[:], softmax_config),
		"softmax output with a single neuron should be invalid",
	)
	testing.expect_value(
		t,
		brain_architecture_error(invalid_softmax_architecture[:], softmax_config),
		"Softmax output activation requires at least two output neurons",
	)
}

@(test)
test_compute_dataset_metrics :: proc(t: ^testing.T) {
	architecture := [2]int{1, 1}
	config := default_brain_config()
	config.output_activation = .Sigmoid
	config.loss = .Binary_Cross_Entropy

	brain := make_brain_with_config(architecture[:], 0.1, config)
	defer destroy_brain(&brain)

	brain.layers[0].neurons[0].out_weights[0] = 10
	brain.layers[1].neurons[0].bias = -5

	raw_inputs := [2][1]f32{{0}, {1}}
	raw_labels := [2][1]f32{{0}, {1}}
	inputs := make([][]f32, len(raw_inputs))
	labels := make([][]f32, len(raw_labels))
	defer delete(inputs)
	defer delete(labels)

	for i := 0; i < len(raw_inputs); i += 1 {
		inputs[i] = raw_inputs[i][:]
		labels[i] = raw_labels[i][:]
	}

	metrics := compute_dataset_metrics(&brain, inputs, labels)
	testing.expect_value(t, metrics.sample_count, 2)
	testing.expect(
		t,
		metrics.accuracy_available,
		"sigmoid output should expose classification accuracy",
	)
	testing.expect(t, math.abs(metrics.accuracy - 1.0) < 1e-6, "accuracy should be 1.0")
	testing.expect(t, metrics.average_loss > 0, "average loss should be positive")
	testing.expect(
		t,
		metrics.average_loss < 0.01,
		"average loss should be low for nearly perfect predictions",
	)
	testing.expect(
		t,
		metrics.mean_squared_error < 0.001,
		"mse should be low for nearly perfect predictions",
	)
	testing.expect(
		t,
		metrics.mean_absolute_error < 0.1,
		"mae should be low for nearly perfect predictions",
	)
}

@(test)
test_compute_multiclass_metrics :: proc(t: ^testing.T) {
	architecture := [2]int{2, 3}
	config := default_brain_config()
	config.output_activation = .Softmax
	config.loss = .Categorical_Cross_Entropy

	brain := make_brain_with_config(architecture[:], 0.1, config)
	defer destroy_brain(&brain)

	brain.layers[0].neurons[0].out_weights[0] = 5
	brain.layers[0].neurons[0].out_weights[1] = 1
	brain.layers[0].neurons[0].out_weights[2] = -1
	brain.layers[0].neurons[1].out_weights[0] = -1
	brain.layers[0].neurons[1].out_weights[1] = 5
	brain.layers[0].neurons[1].out_weights[2] = 1
	brain.layers[1].neurons[0].bias = 0
	brain.layers[1].neurons[1].bias = 0
	brain.layers[1].neurons[2].bias = -2

	raw_inputs := [3][2]f32{{1, 0}, {0, 1}, {1, 1}}
	raw_labels := [3][3]f32{{1, 0, 0}, {0, 1, 0}, {0, 1, 0}}
	inputs := make([][]f32, len(raw_inputs))
	labels := make([][]f32, len(raw_labels))
	defer delete(inputs)
	defer delete(labels)

	for i := 0; i < len(raw_inputs); i += 1 {
		inputs[i] = raw_inputs[i][:]
		labels[i] = raw_labels[i][:]
	}

	metrics := compute_dataset_metrics(&brain, inputs, labels)
	testing.expect_value(t, metrics.sample_count, 3)
	testing.expect(
		t,
		metrics.accuracy_available,
		"softmax output should expose multiclass accuracy",
	)
	testing.expect(
		t,
		math.abs(metrics.accuracy - 1.0) < 1e-6,
		"softmax multiclass accuracy should be 1.0",
	)
	testing.expect(
		t,
		metrics.average_loss < 0.5,
		"softmax loss should be low for confident correct predictions",
	)

	report, ok := compute_classification_report(&brain, inputs, labels)
	if !testing.expect(t, ok, "classification report should be available for softmax") {
		return
	}
	defer destroy_classification_report(&report)

	testing.expect_value(t, report.class_count, 3)
	testing.expect_value(t, report.sample_count, 3)
	testing.expect_value(t, report.correct_count, 3)
	testing.expect(t, report.labels_are_valid, "softmax labels should be considered valid")
	testing.expect_value(t, report.confusion_matrix[0], 1)
	testing.expect_value(t, report.confusion_matrix[4], 2)
	testing.expect_value(t, report.confusion_matrix[8], 0)
}

@(test)
test_validate_label_vector :: proc(t: ^testing.T) {
	binary_config := default_brain_config()
	binary_config.output_activation = .Sigmoid
	binary_config.loss = .Binary_Cross_Entropy
	valid_binary := [1]f32{1}
	invalid_binary := [1]f32{1.5}

	testing.expect(
		t,
		validate_label_vector(binary_config, valid_binary[:]),
		"binary cross entropy labels in [0, 1] should be valid",
	)
	testing.expect(
		t,
		!validate_label_vector(binary_config, invalid_binary[:]),
		"binary cross entropy labels outside [0, 1] should be invalid",
	)
	testing.expect_value(
		t,
		label_vector_error(binary_config, invalid_binary[:]),
		"Binary_Cross_Entropy labels must be in the range [0, 1]",
	)

	softmax_config := default_brain_config()
	softmax_config.output_activation = .Softmax
	softmax_config.loss = .Categorical_Cross_Entropy
	valid_distribution := [3]f32{0, 1, 0}
	invalid_distribution := [3]f32{0.5, 0.5, 0.5}

	testing.expect(
		t,
		validate_label_vector(softmax_config, valid_distribution[:]),
		"categorical labels that sum to 1 should be valid",
	)
	testing.expect(
		t,
		!validate_label_vector(softmax_config, invalid_distribution[:]),
		"categorical labels that do not sum to 1 should be invalid",
	)
	testing.expect_value(
		t,
		label_vector_error(softmax_config, invalid_distribution[:]),
		"Categorical_Cross_Entropy labels must sum to 1.0",
	)
}

@(test)
test_train_with_progress_callback :: proc(t: ^testing.T) {
	callback_invocations = 0

	architecture := [3]int{2, 2, 1}
	config := default_brain_config()
	config.output_activation = .Sigmoid
	config.loss = .Binary_Cross_Entropy

	brain := make_brain_with_config(architecture[:], 0.15, config)
	defer destroy_brain(&brain)

	raw_inputs := [4][2]f32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	raw_labels := [4][1]f32{{0}, {1}, {1}, {0}}
	inputs := make([][]f32, len(raw_inputs))
	labels := make([][]f32, len(raw_labels))
	defer delete(inputs)
	defer delete(labels)

	for i := 0; i < len(raw_inputs); i += 1 {
		inputs[i] = raw_inputs[i][:]
		labels[i] = raw_labels[i][:]
	}

	training := default_training_config()
	training.batch_size = 4
	training.report_every = 5
	training.progress_callback = record_progress_callback

	train_with_config(&brain, inputs, labels, 12, training)
	testing.expect_value(t, callback_invocations, 3)
}
