package nn

import json "core:encoding/json"
import "core:os"

Model_File_Neuron :: struct {
	bias:        f32,
	out_weights: []f32,
}

Model_File_Layer :: struct {
	neurons: []Model_File_Neuron,
}

Model_File :: struct {
	version:      int,
	alpha:        f32,
	config:       Brain_Config,
	architecture: []int,
	layers:       []Model_File_Layer,
}

MODEL_FILE_VERSION :: 1

save_brain :: proc(brain: ^Brain, path: string) -> bool {
	model_file := model_file_from_brain(brain)
	defer destroy_model_file(&model_file)
	return save_model_file(&model_file, path)
}

load_brain :: proc(path: string) -> (brain: Brain, ok: bool) {
	model_file, loaded := load_model_file(path)
	if !loaded {
		return
	}
	defer destroy_model_file(&model_file)

	brain, ok = brain_from_model_file(&model_file)
	return
}

save_model_file :: proc(model_file: ^Model_File, path: string) -> bool {
	data, marshal_err := json.marshal(model_file^)
	if marshal_err != nil {
		return false
	}
	defer delete(data)

	write_err := os.write_entire_file(path, data)
	return write_err == nil
}

load_model_file :: proc(path: string) -> (model_file: Model_File, ok: bool) {
	if !os.exists(path) {
		return
	}

	data, read_err := os.read_entire_file(path, context.allocator)
	if read_err != nil {
		return
	}
	defer delete(data)

	unmarshal_err := json.unmarshal(data, &model_file)
	if unmarshal_err != nil {
		destroy_model_file(&model_file)
		return
	}

	if !validate_model_file(&model_file) {
		destroy_model_file(&model_file)
		return
	}

	ok = true
	return
}

model_file_from_brain :: proc(brain: ^Brain) -> (model_file: Model_File) {
	model_file.version = MODEL_FILE_VERSION
	model_file.alpha = brain.alpha
	model_file.config = brain.config
	model_file.architecture = make([]int, len(brain.layers))
	model_file.layers = make([]Model_File_Layer, len(brain.layers))

	for layer_index := 0; layer_index < len(brain.layers); layer_index += 1 {
		model_file.architecture[layer_index] = len(brain.layers[layer_index].neurons)
		model_file.layers[layer_index].neurons = make(
			[]Model_File_Neuron,
			len(brain.layers[layer_index].neurons),
		)

		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			neuron := brain.layers[layer_index].neurons[neuron_index]
			model_neuron := &model_file.layers[layer_index].neurons[neuron_index]
			model_neuron.bias = neuron.bias
			model_neuron.out_weights = make([]f32, len(neuron.out_weights))

			for weight_index := 0; weight_index < len(neuron.out_weights); weight_index += 1 {
				model_neuron.out_weights[weight_index] = neuron.out_weights[weight_index]
			}
		}
	}

	return
}

brain_from_model_file :: proc(model_file: ^Model_File) -> (brain: Brain, ok: bool) {
	if !validate_model_file(model_file) {
		return
	}

	brain = make_brain_with_config(model_file.architecture, model_file.alpha, model_file.config)
	if !restore_brain_from_model_file(&brain, model_file) {
		destroy_brain(&brain)
		brain = {}
		return
	}

	ok = true
	return
}

restore_brain_from_model_file :: proc(brain: ^Brain, model_file: ^Model_File) -> bool {
	if len(brain.layers) != len(model_file.layers) {
		return false
	}

	for layer_index := 0; layer_index < len(brain.layers); layer_index += 1 {
		if len(brain.layers[layer_index].neurons) != len(model_file.layers[layer_index].neurons) {
			return false
		}

		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			neuron := &brain.layers[layer_index].neurons[neuron_index]
			model_neuron := model_file.layers[layer_index].neurons[neuron_index]

			if len(neuron.out_weights) != len(model_neuron.out_weights) {
				return false
			}

			neuron.bias = model_neuron.bias
			for weight_index := 0; weight_index < len(neuron.out_weights); weight_index += 1 {
				neuron.out_weights[weight_index] = model_neuron.out_weights[weight_index]
			}
		}
	}

	return true
}

validate_model_file :: proc(model_file: ^Model_File) -> bool {
	if model_file.version != MODEL_FILE_VERSION {
		return false
	}

	if !validate_brain_config(model_file.config) {
		return false
	}

	if !validate_brain_architecture(model_file.architecture, model_file.config) {
		return false
	}

	if len(model_file.architecture) < 2 || len(model_file.layers) != len(model_file.architecture) {
		return false
	}

	for layer_index := 0; layer_index < len(model_file.architecture); layer_index += 1 {
		if model_file.architecture[layer_index] <= 0 {
			return false
		}

		if len(model_file.layers[layer_index].neurons) != model_file.architecture[layer_index] {
			return false
		}

		expected_out_weights := 0
		if layer_index + 1 < len(model_file.architecture) {
			expected_out_weights = model_file.architecture[layer_index + 1]
		}

		for neuron_index := 0;
		    neuron_index < len(model_file.layers[layer_index].neurons);
		    neuron_index += 1 {
			if len(model_file.layers[layer_index].neurons[neuron_index].out_weights) !=
			   expected_out_weights {
				return false
			}
		}
	}

	return true
}

destroy_model_file :: proc(model_file: ^Model_File) {
	for layer_index := 0; layer_index < len(model_file.layers); layer_index += 1 {
		for neuron_index := 0;
		    neuron_index < len(model_file.layers[layer_index].neurons);
		    neuron_index += 1 {
			delete(model_file.layers[layer_index].neurons[neuron_index].out_weights)
		}

		delete(model_file.layers[layer_index].neurons)
	}

	delete(model_file.architecture)
	delete(model_file.layers)
	model_file^ = {}
}
