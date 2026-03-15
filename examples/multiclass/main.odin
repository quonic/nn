package multiclass_example

import nn "../.."
import "core:fmt"
import "core:math/rand"

print_training_progress :: proc(progress: nn.Training_Progress) {
	metrics := progress.metrics
	fmt.printf(
		"epoch=%d/%d loss=%0.5f accuracy=%0.2f%%\n",
		progress.epoch,
		progress.total_epochs,
		metrics.average_loss,
		metrics.accuracy * 100,
	)
}

main :: proc() {
	architecture := [3]int{2, 6, 3}
	config := nn.default_brain_config()
	config.hidden_activation = .Tanh
	config.output_activation = .Softmax
	config.loss = .Categorical_Cross_Entropy

	rand.reset_u64(7)
	brain := nn.make_brain_with_config(architecture[:], 0.12, config)
	defer nn.destroy_brain(&brain)

	raw_inputs := [9][2]f32 {
		{-1.0, -0.8},
		{-0.9, -1.1},
		{-1.2, -0.9},
		{1.0, -0.9},
		{0.8, -1.2},
		{1.1, -0.7},
		{0.0, 1.0},
		{-0.2, 1.2},
		{0.2, 0.8},
	}
	raw_labels := [9][3]f32 {
		{1, 0, 0},
		{1, 0, 0},
		{1, 0, 0},
		{0, 1, 0},
		{0, 1, 0},
		{0, 1, 0},
		{0, 0, 1},
		{0, 0, 1},
		{0, 0, 1},
	}

	inputs := make([][]f32, len(raw_inputs))
	labels := make([][]f32, len(raw_labels))
	defer delete(inputs)
	defer delete(labels)

	for i := 0; i < len(raw_inputs); i += 1 {
		inputs[i] = raw_inputs[i][:]
		labels[i] = raw_labels[i][:]
	}

	training := nn.default_training_config()
	training.batch_size = 3
	training.shuffle_samples = true
	training.report_every = 500
	training.progress_callback = print_training_progress

	fmt.println("Training multiclass toy model...")
	nn.train_with_config(&brain, inputs, labels, 4_000, training)

	metrics := nn.compute_dataset_metrics(&brain, inputs, labels)
	fmt.printf(
		"final loss=%0.5f mse=%0.5f mae=%0.5f accuracy=%0.2f%%\n",
		metrics.average_loss,
		metrics.mean_squared_error,
		metrics.mean_absolute_error,
		metrics.accuracy * 100,
	)

	report, ok := nn.compute_classification_report(&brain, inputs, labels)
	if ok {
		defer nn.destroy_classification_report(&report)
		fmt.println("confusion matrix (rows=expected, cols=predicted):")
		for row := 0; row < report.class_count; row += 1 {
			for col := 0; col < report.class_count; col += 1 {
				if col > 0 {
					fmt.print(" ")
				}
				fmt.printf("%d", report.confusion_matrix[row * report.class_count + col])
			}
			fmt.println()
		}
	}

	fmt.println("sample predictions:")
	for i := 0; i < len(raw_inputs); i += 1 {
		output := nn.run(&brain, raw_inputs[i][:])
		predicted_class := 0
		predicted_value := output[0]
		for j := 1; j < len(output); j += 1 {
			if output[j] > predicted_value {
				predicted_value = output[j]
				predicted_class = j
			}
		}

		expected_class := 0
		for j := 1; j < len(raw_labels[i]); j += 1 {
			if raw_labels[i][j] > raw_labels[i][expected_class] {
				expected_class = j
			}
		}

		fmt.printf(
			"input=%v expected=%d predicted=%d output=%v\n",
			raw_inputs[i],
			expected_class,
			predicted_class,
			output,
		)

		delete(output)
	}
}
