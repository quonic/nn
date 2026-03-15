package nn

import "core:os"
import "core:testing"

@(test)
test_model_file_round_trip :: proc(t: ^testing.T) {
	path := "nn-roundtrip-test.json"
	_ = os.remove(path)
	defer os.remove(path)

	architecture := [3]int{2, 3, 1}
	config := default_brain_config()
	config.hidden_activation = .Tanh
	config.output_activation = .Linear
	config.loss = .Mean_Absolute_Error

	brain := make_brain_with_config(architecture[:], 0.25, config)
	defer destroy_brain(&brain)
	assign_test_parameters(&brain)

	ok := save_brain(&brain, path)
	if !testing.expect(t, ok, "save_brain should succeed") {
		return
	}

	model_file, loaded_model := load_model_file(path)
	if !testing.expect(t, loaded_model, "load_model_file should succeed") {
		return
	}
	defer destroy_model_file(&model_file)

	testing.expect_value(t, model_file.version, MODEL_FILE_VERSION)
	testing.expect_value(t, model_file.alpha, f32(0.25))
	testing.expect_value(t, model_file.config.hidden_activation, Activation_Kind.Tanh)
	testing.expect_value(t, model_file.config.output_activation, Activation_Kind.Linear)
	testing.expect_value(t, model_file.config.loss, Loss_Kind.Mean_Absolute_Error)

	loaded_brain, loaded_brain_ok := load_brain(path)
	if !testing.expect(t, loaded_brain_ok, "load_brain should succeed") {
		return
	}
	defer destroy_brain(&loaded_brain)

	compare_brains_exact(t, &brain, &loaded_brain)

	sample := [2]f32{0.25, -0.5}
	original_output := run(&brain, sample[:])
	defer delete(original_output)
	loaded_output := run(&loaded_brain, sample[:])
	defer delete(loaded_output)

	testing.expect_value(t, len(original_output), len(loaded_output))
	for i := 0; i < len(original_output); i += 1 {
		testing.expect_value(t, original_output[i], loaded_output[i])
	}
}

assign_test_parameters :: proc(brain: ^Brain) {
	value := f32(-0.375)
	for layer_index := 0; layer_index < len(brain.layers); layer_index += 1 {
		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			neuron := &brain.layers[layer_index].neurons[neuron_index]
			neuron.bias = value
			value += 0.125

			for weight_index := 0; weight_index < len(neuron.out_weights); weight_index += 1 {
				neuron.out_weights[weight_index] = value
				value += 0.125
			}
		}
	}
}

compare_brains_exact :: proc(t: ^testing.T, left, right: ^Brain) {
	testing.expect_value(t, left.alpha, right.alpha)
	testing.expect_value(t, left.config.hidden_activation, right.config.hidden_activation)
	testing.expect_value(t, left.config.output_activation, right.config.output_activation)
	testing.expect_value(t, left.config.loss, right.config.loss)
	testing.expect_value(t, len(left.layers), len(right.layers))

	for layer_index := 0; layer_index < len(left.layers); layer_index += 1 {
		testing.expect_value(
			t,
			len(left.layers[layer_index].neurons),
			len(right.layers[layer_index].neurons),
		)

		for neuron_index := 0;
		    neuron_index < len(left.layers[layer_index].neurons);
		    neuron_index += 1 {
			left_neuron := left.layers[layer_index].neurons[neuron_index]
			right_neuron := right.layers[layer_index].neurons[neuron_index]
			testing.expect_value(t, left_neuron.bias, right_neuron.bias)
			testing.expect_value(t, len(left_neuron.out_weights), len(right_neuron.out_weights))

			for weight_index := 0; weight_index < len(left_neuron.out_weights); weight_index += 1 {
				testing.expect_value(
					t,
					left_neuron.out_weights[weight_index],
					right_neuron.out_weights[weight_index],
				)
			}
		}
	}
}
