package nn

import "core:math"
import "core:math/rand"

Activation_Kind :: enum {
	ReLU,
	Leaky_ReLU,
	Sigmoid,
	Softmax,
	Tanh,
	Linear,
}

Loss_Kind :: enum {
	Mean_Squared_Error,
	Mean_Absolute_Error,
	Binary_Cross_Entropy,
	Categorical_Cross_Entropy,
}

Brain_Config :: struct {
	hidden_activation: Activation_Kind,
	output_activation: Activation_Kind,
	loss:              Loss_Kind,
}

Training_Config :: struct {
	batch_size:        int,
	shuffle_samples:   bool,
	report_every:      int,
	progress_callback: Training_Progress_Callback,
}

Training_Progress :: struct {
	epoch:        int,
	total_epochs: int,
	metrics:      Dataset_Metrics,
}

Training_Progress_Callback :: proc(progress: Training_Progress)

Dataset_Metrics :: struct {
	sample_count:        int,
	average_loss:        f32,
	mean_absolute_error: f32,
	mean_squared_error:  f32,
	accuracy:            f32,
	accuracy_available:  bool,
}

Classification_Report :: struct {
	class_count:      int,
	sample_count:     int,
	correct_count:    int,
	accuracy:         f32,
	confusion_matrix: []int,
	row_major_labels: bool,
	labels_are_valid: bool,
}

Brain :: struct {
	layers: []Layer,
	alpha:  f32,
	config: Brain_Config,
}

make_brain :: proc(layer_sizes: []int, alpha: f32) -> (brain: Brain) {
	return make_brain_with_config(layer_sizes, alpha, default_brain_config())
}

default_brain_config :: proc() -> (config: Brain_Config) {
	config.hidden_activation = .ReLU
	config.output_activation = .Sigmoid
	config.loss = .Mean_Squared_Error
	return
}

default_training_config :: proc() -> (config: Training_Config) {
	config.batch_size = 1
	config.shuffle_samples = false
	config.report_every = 0
	config.progress_callback = nil
	return
}

validate_brain_config :: proc(config: Brain_Config) -> bool {
	return brain_config_error(config) == ""
}

validate_brain_architecture :: proc(layer_sizes: []int, config: Brain_Config) -> bool {
	return brain_architecture_error(layer_sizes, config) == ""
}

brain_config_error :: proc(config: Brain_Config) -> string {
	switch config.loss {
	case .Binary_Cross_Entropy:
		if config.output_activation != .Sigmoid {
			return "Binary_Cross_Entropy requires Sigmoid output activation"
		}
		return ""
	case .Categorical_Cross_Entropy:
		if config.output_activation != .Softmax {
			return "Categorical_Cross_Entropy requires Softmax output activation"
		}
		return ""
	case .Mean_Squared_Error, .Mean_Absolute_Error:
		if config.output_activation == .Softmax {
			return "Softmax output activation requires Categorical_Cross_Entropy"
		}
		return ""
	}

	return "Unsupported brain configuration"
}

brain_architecture_error :: proc(layer_sizes: []int, config: Brain_Config) -> string {
	if len(layer_sizes) < 2 {
		return "Brain architecture requires at least an input and an output layer"
	}

	for i := 0; i < len(layer_sizes); i += 1 {
		if layer_sizes[i] <= 0 {
			return "Brain architecture requires every layer to contain at least one neuron"
		}
	}

	output_width := layer_sizes[len(layer_sizes) - 1]
	if config.output_activation == .Softmax && output_width < 2 {
		return "Softmax output activation requires at least two output neurons"
	}

	if config.loss == .Categorical_Cross_Entropy && output_width < 2 {
		return "Categorical_Cross_Entropy requires at least two output neurons"
	}

	return ""
}

make_brain_with_config :: proc(
	layer_sizes: []int,
	alpha: f32,
	config: Brain_Config,
) -> (
	brain: Brain,
) {
	assert(len(layer_sizes) >= 2)
	assert(validate_brain_config(config), brain_config_error(config))
	assert(
		validate_brain_architecture(layer_sizes, config),
		brain_architecture_error(layer_sizes, config),
	)

	brain.alpha = alpha
	brain.config = config
	brain.layers = make([]Layer, len(layer_sizes))

	for i := 0; i < len(layer_sizes); i += 1 {
		next_layer_size := 0
		if i + 1 < len(layer_sizes) {
			next_layer_size = layer_sizes[i + 1]
		}

		brain.layers[i] = make_layer(layer_sizes[i], next_layer_size)
	}

	initialize_weights(&brain)
	return
}

destroy_brain :: proc(brain: ^Brain) {
	for i := 0; i < len(brain.layers); i += 1 {
		destroy_layer(&brain.layers[i])
	}

	delete(brain.layers)
	brain^ = {}
}

train :: proc(brain: ^Brain, inputs, labels: [][]f32, epochs: int) {
	train_with_config(brain, inputs, labels, epochs, default_training_config())
}

train_with_config :: proc(
	brain: ^Brain,
	inputs, labels: [][]f32,
	epochs: int,
	config: Training_Config,
) {
	assert(validate_brain_config(brain.config), brain_config_error(brain.config))
	assert(brain_runtime_architecture_error(brain) == "", brain_runtime_architecture_error(brain))
	assert(len(inputs) == len(labels))

	if len(inputs) == 0 {
		return
	}

	for i := 0; i < len(inputs); i += 1 {
		assert(len(inputs[i]) == len(brain.layers[0].neurons))
		assert(len(labels[i]) == len(brain.layers[len(brain.layers) - 1].neurons))
		assert(
			validate_label_vector(brain.config, labels[i]),
			label_vector_error(brain.config, labels[i]),
		)
	}

	batch_size := config.batch_size
	if batch_size <= 0 {
		batch_size = 1
	}
	if batch_size > len(inputs) {
		batch_size = len(inputs)
	}

	report_every := config.report_every
	if report_every < 0 {
		report_every = 0
	}

	sample_order := make([]int, len(inputs))
	defer delete(sample_order)
	for i := 0; i < len(sample_order); i += 1 {
		sample_order[i] = i
	}

	for epoch := 0; epoch < epochs; epoch += 1 {
		if config.shuffle_samples {
			rand.shuffle(sample_order)
		}

		for batch_start := 0; batch_start < len(inputs); batch_start += batch_size {
			batch_end := batch_start + batch_size
			if batch_end > len(inputs) {
				batch_end = len(inputs)
			}

			zero_batch_gradients(brain)

			for batch_index := batch_start; batch_index < batch_end; batch_index += 1 {
				sample_index := sample_order[batch_index]
				feed_input(brain, inputs[sample_index])
				forward_prop(brain)
				back_prop(brain, labels[sample_index])
			}

			update_weights(brain, batch_end - batch_start)
		}

		if config.progress_callback != nil &&
		   (report_every > 0 && ((epoch + 1) % report_every) == 0 || epoch + 1 == epochs) {
			config.progress_callback(
				Training_Progress {
					epoch = epoch + 1,
					total_epochs = epochs,
					metrics = compute_dataset_metrics(brain, inputs, labels),
				},
			)
		}
	}
}

compute_classification_report :: proc(
	brain: ^Brain,
	inputs, labels: [][]f32,
) -> (
	report: Classification_Report,
	ok: bool,
) {
	assert(len(inputs) == len(labels))
	if !classification_metrics_available(brain.config) {
		return
	}

	report.sample_count = len(inputs)
	report.row_major_labels = true
	report.labels_are_valid = true
	if brain.config.output_activation == .Softmax {
		report.class_count = len(brain.layers[len(brain.layers) - 1].neurons)
	} else {
		report.class_count = 2
	}
	report.confusion_matrix = make([]int, report.class_count * report.class_count)

	for sample_index := 0; sample_index < len(inputs); sample_index += 1 {
		if !validate_label_vector(brain.config, labels[sample_index]) {
			report.labels_are_valid = false
			continue
		}

		feed_input(brain, inputs[sample_index])
		forward_prop(brain)

		expected_class := expected_class_for_label(brain.config, labels[sample_index])
		actual_class := predicted_class_for_brain(brain)
		if expected_class < 0 ||
		   actual_class < 0 ||
		   expected_class >= report.class_count ||
		   actual_class >= report.class_count {
			report.labels_are_valid = false
			continue
		}

		report.confusion_matrix[expected_class * report.class_count + actual_class] += 1
		if expected_class == actual_class {
			report.correct_count += 1
		}
	}

	if report.sample_count > 0 {
		report.accuracy = f32(report.correct_count) / f32(report.sample_count)
	}

	ok = true
	return
}

destroy_classification_report :: proc(report: ^Classification_Report) {
	delete(report.confusion_matrix)
	report^ = {}
}

compute_dataset_metrics :: proc(
	brain: ^Brain,
	inputs, labels: [][]f32,
) -> (
	metrics: Dataset_Metrics,
) {
	assert(len(inputs) == len(labels))
	metrics.sample_count = len(inputs)
	metrics.accuracy_available = classification_metrics_available(brain.config)

	if len(inputs) == 0 {
		return
	}

	matched_samples := 0
	for sample_index := 0; sample_index < len(inputs); sample_index += 1 {
		feed_input(brain, inputs[sample_index])
		forward_prop(brain)

		last_layer := brain.layers[len(brain.layers) - 1]
		sample_matches := true
		actual_class_index := 0
		expected_class_index := 0
		actual_class_value := f32(0)
		expected_class_value := f32(0)
		for output_index := 0; output_index < len(last_layer.neurons); output_index += 1 {
			actual := last_layer.neurons[output_index].actv
			expected := labels[sample_index][output_index]
			delta := expected - actual

			metrics.average_loss += loss_value(brain.config.loss, expected, actual)
			metrics.mean_absolute_error += abs_f32(delta)
			metrics.mean_squared_error += delta * delta

			if metrics.accuracy_available {
				if brain.config.output_activation == .Softmax {
					if output_index == 0 || actual > actual_class_value {
						actual_class_value = actual
						actual_class_index = output_index
					}
					if output_index == 0 || expected > expected_class_value {
						expected_class_value = expected
						expected_class_index = output_index
					}
				} else if classify_output(brain.config.output_activation, actual) !=
				   classify_output(brain.config.output_activation, expected) {
					sample_matches = false
				}
			}
		}

		if metrics.accuracy_available && brain.config.output_activation == .Softmax {
			sample_matches = actual_class_index == expected_class_index
		}

		if metrics.accuracy_available && sample_matches {
			matched_samples += 1
		}
	}

	sample_count_f32 := f32(len(inputs))
	metrics.average_loss /= sample_count_f32
	metrics.mean_absolute_error /= sample_count_f32
	metrics.mean_squared_error /= sample_count_f32
	if metrics.accuracy_available {
		metrics.accuracy = f32(matched_samples) / sample_count_f32
	}

	return
}

run :: proc(brain: ^Brain, input: []f32) -> (output: []f32) {
	feed_input(brain, input)
	forward_prop(brain)

	last_layer := brain.layers[len(brain.layers) - 1]
	output = make([]f32, len(last_layer.neurons))
	for i := 0; i < len(last_layer.neurons); i += 1 {
		output[i] = last_layer.neurons[i].actv
	}

	return
}

brain_runtime_architecture_error :: proc(brain: ^Brain) -> string {
	if len(brain.layers) < 2 {
		return "Brain architecture requires at least an input and an output layer"
	}

	for i := 0; i < len(brain.layers); i += 1 {
		if len(brain.layers[i].neurons) <= 0 {
			return "Brain architecture requires every layer to contain at least one neuron"
		}
	}

	output_width := len(brain.layers[len(brain.layers) - 1].neurons)
	if brain.config.output_activation == .Softmax && output_width < 2 {
		return "Softmax output activation requires at least two output neurons"
	}

	if brain.config.loss == .Categorical_Cross_Entropy && output_width < 2 {
		return "Categorical_Cross_Entropy requires at least two output neurons"
	}

	return ""
}

compute_cost :: proc(brain: ^Brain, expected: []f32) -> (cost: f32) {
	last_layer := brain.layers[len(brain.layers) - 1]
	assert(len(expected) == len(last_layer.neurons))

	for i := 0; i < len(last_layer.neurons); i += 1 {
		cost += loss_value(brain.config.loss, expected[i], last_layer.neurons[i].actv)
	}

	return
}

initialize_weights :: proc(brain: ^Brain) {
	for layer_index := 0; layer_index < len(brain.layers) - 1; layer_index += 1 {
		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			neuron := &brain.layers[layer_index].neurons[neuron_index]

			for weight_index := 0; weight_index < len(neuron.out_weights); weight_index += 1 {
				neuron.out_weights[weight_index] = rand.float32()
			}

			if layer_index > 0 {
				neuron.bias = rand.float32()
			}
		}
	}

	last_layer := &brain.layers[len(brain.layers) - 1]
	for neuron_index := 0; neuron_index < len(last_layer.neurons); neuron_index += 1 {
		last_layer.neurons[neuron_index].bias = rand.float32()
	}
}

feed_input :: proc(brain: ^Brain, input: []f32) {
	assert(len(input) == len(brain.layers[0].neurons))

	for i := 0; i < len(input); i += 1 {
		brain.layers[0].neurons[i].actv = input[i]
	}
}

forward_prop :: proc(brain: ^Brain) {
	last_index := len(brain.layers) - 1

	for layer_index := 1; layer_index < len(brain.layers); layer_index += 1 {
		is_output_layer := layer_index == last_index
		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			neuron := &brain.layers[layer_index].neurons[neuron_index]
			neuron.z = neuron.bias

			for previous_index := 0;
			    previous_index < len(brain.layers[layer_index - 1].neurons);
			    previous_index += 1 {
				previous_neuron := brain.layers[layer_index - 1].neurons[previous_index]
				neuron.z += previous_neuron.actv * previous_neuron.out_weights[neuron_index]
			}
		}

		if is_output_layer {
			if brain.config.output_activation == .Softmax {
				apply_softmax(&brain.layers[layer_index])
			} else {
				for neuron_index := 0;
				    neuron_index < len(brain.layers[layer_index].neurons);
				    neuron_index += 1 {
					neuron := &brain.layers[layer_index].neurons[neuron_index]
					neuron.actv = activation_value(brain.config.output_activation, neuron.z)
				}
			}
		} else {
			for neuron_index := 0;
			    neuron_index < len(brain.layers[layer_index].neurons);
			    neuron_index += 1 {
				neuron := &brain.layers[layer_index].neurons[neuron_index]
				neuron.actv = activation_value(brain.config.hidden_activation, neuron.z)
			}
		}
	}
}

back_prop :: proc(brain: ^Brain, expected: []f32) {
	last_index := len(brain.layers) - 1
	assert(len(expected) == len(brain.layers[last_index].neurons))

	zero_backprop_state(brain)

	for neuron_index := 0;
	    neuron_index < len(brain.layers[last_index].neurons);
	    neuron_index += 1 {
		output_neuron := &brain.layers[last_index].neurons[neuron_index]
		output_neuron.dz = output_delta(brain, expected[neuron_index], output_neuron)
		output_neuron.dbias += output_neuron.dz

		for previous_index := 0;
		    previous_index < len(brain.layers[last_index - 1].neurons);
		    previous_index += 1 {
			previous_neuron := &brain.layers[last_index - 1].neurons[previous_index]
			previous_neuron.dw[neuron_index] += previous_neuron.actv * output_neuron.dz
			previous_neuron.dactv += previous_neuron.out_weights[neuron_index] * output_neuron.dz
		}
	}

	for layer_index := last_index - 1; layer_index > 0; layer_index -= 1 {
		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			neuron := &brain.layers[layer_index].neurons[neuron_index]
			neuron.dz =
				neuron.dactv *
				activation_derivative(brain.config.hidden_activation, neuron.z, neuron.actv)
			neuron.dbias += neuron.dz

			for previous_index := 0;
			    previous_index < len(brain.layers[layer_index - 1].neurons);
			    previous_index += 1 {
				previous_neuron := &brain.layers[layer_index - 1].neurons[previous_index]
				previous_neuron.dw[neuron_index] += previous_neuron.actv * neuron.dz

				if layer_index > 1 {
					previous_neuron.dactv += previous_neuron.out_weights[neuron_index] * neuron.dz
				}
			}
		}
	}
}

update_weights :: proc(brain: ^Brain, batch_sample_count: int) {
	scale := brain.alpha / f32(batch_sample_count)

	for layer_index := 0; layer_index < len(brain.layers) - 1; layer_index += 1 {
		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			neuron := &brain.layers[layer_index].neurons[neuron_index]
			for weight_index := 0; weight_index < len(neuron.out_weights); weight_index += 1 {
				neuron.out_weights[weight_index] -= scale * neuron.dw[weight_index]
			}
		}
	}

	for layer_index := 1; layer_index < len(brain.layers); layer_index += 1 {
		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			brain.layers[layer_index].neurons[neuron_index].bias -=
				scale * brain.layers[layer_index].neurons[neuron_index].dbias
		}
	}
}

zero_batch_gradients :: proc(brain: ^Brain) {
	for layer_index := 0; layer_index < len(brain.layers); layer_index += 1 {
		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			neuron := &brain.layers[layer_index].neurons[neuron_index]
			neuron.dbias = 0

			for weight_index := 0; weight_index < len(neuron.dw); weight_index += 1 {
				neuron.dw[weight_index] = 0
			}
		}
	}
}

zero_backprop_state :: proc(brain: ^Brain) {
	for layer_index := 0; layer_index < len(brain.layers); layer_index += 1 {
		for neuron_index := 0;
		    neuron_index < len(brain.layers[layer_index].neurons);
		    neuron_index += 1 {
			neuron := &brain.layers[layer_index].neurons[neuron_index]
			neuron.dactv = 0
			neuron.dz = 0
		}
	}
}

activation_value :: proc(kind: Activation_Kind, x: f32) -> f32 {
	switch kind {
	case .ReLU:
		if x > 0 {
			return x
		}
		return 0
	case .Leaky_ReLU:
		if x > 0 {
			return x
		}
		return 0.01 * x
	case .Sigmoid:
		return 1 / (1 + f32(math.exp(f64(-x))))
	case .Softmax:
		return x
	case .Tanh:
		return math.tanh(x)
	case .Linear:
		return x
	}

	return 0
}

activation_derivative :: proc(kind: Activation_Kind, x, output: f32) -> f32 {
	switch kind {
	case .ReLU:
		if x > 0 {
			return 1
		}
		return 0
	case .Leaky_ReLU:
		if x > 0 {
			return 1
		}
		return 0.01
	case .Sigmoid:
		return output * (1 - output)
	case .Softmax:
		return 1
	case .Tanh:
		return 1 - output * output
	case .Linear:
		return 1
	}

	return 0
}
loss_value :: proc(kind: Loss_Kind, expected, actual: f32) -> f32 {
	switch kind {
	case .Mean_Squared_Error:
		delta := expected - actual
		return 0.5 * delta * delta
	case .Mean_Absolute_Error:
		return abs_f32(expected - actual)
	case .Binary_Cross_Entropy:
		clamped_actual := clamp_probability(actual)
		return -(expected * math.ln(clamped_actual) + (1 - expected) * math.ln(1 - clamped_actual))
	case .Categorical_Cross_Entropy:
		if expected <= 0 {
			return 0
		}
		return -expected * math.ln(clamp_probability(actual))
	}

	return 0
}

loss_derivative :: proc(kind: Loss_Kind, expected, actual: f32) -> f32 {
	switch kind {
	case .Mean_Squared_Error:
		return actual - expected
	case .Mean_Absolute_Error:
		delta := actual - expected
		if delta > 0 {
			return 1
		}
		if delta < 0 {
			return -1
		}
		return 0
	case .Binary_Cross_Entropy:
		clamped_actual := clamp_probability(actual)
		return (clamped_actual - expected) / (clamped_actual * (1 - clamped_actual))
	case .Categorical_Cross_Entropy:
		return -expected / clamp_probability(actual)
	}

	return 0
}

output_delta :: proc(brain: ^Brain, expected: f32, neuron: ^Neuron) -> f32 {
	if brain.config.loss == .Binary_Cross_Entropy && brain.config.output_activation == .Sigmoid {
		return neuron.actv - expected
	}
	if brain.config.loss == .Categorical_Cross_Entropy &&
	   brain.config.output_activation == .Softmax {
		return neuron.actv - expected
	}

	return(
		loss_derivative(brain.config.loss, expected, neuron.actv) *
		activation_derivative(brain.config.output_activation, neuron.z, neuron.actv) \
	)
}

classification_metrics_available :: proc(config: Brain_Config) -> bool {
	return(
		config.output_activation == .Sigmoid ||
		config.output_activation == .Tanh ||
		config.output_activation == .Softmax \
	)
}

validate_label_vector :: proc(config: Brain_Config, label: []f32) -> bool {
	return label_vector_error(config, label) == ""
}

label_vector_error :: proc(config: Brain_Config, label: []f32) -> string {
	switch config.loss {
	case .Binary_Cross_Entropy:
		for i := 0; i < len(label); i += 1 {
			if label[i] < 0 || label[i] > 1 {
				return "Binary_Cross_Entropy labels must be in the range [0, 1]"
			}
		}
		return ""
	case .Categorical_Cross_Entropy:
		if len(label) < 2 {
			return "Categorical_Cross_Entropy labels require at least two classes"
		}

		sum := f32(0)
		for i := 0; i < len(label); i += 1 {
			if label[i] < 0 || label[i] > 1 {
				return "Categorical_Cross_Entropy labels must be probabilities in the range [0, 1]"
			}
			sum += label[i]
		}

		if abs_f32(sum - 1) > 1e-4 {
			return "Categorical_Cross_Entropy labels must sum to 1.0"
		}
		return ""
	case .Mean_Squared_Error, .Mean_Absolute_Error:
		return ""
	}

	return "Unsupported label vector"
}

classify_output :: proc(kind: Activation_Kind, value: f32) -> int {
	switch kind {
	case .Sigmoid:
		if value >= 0.5 {
			return 1
		}
		return 0
	case .Tanh:
		if value >= 0 {
			return 1
		}
		return 0
	case .ReLU, .Leaky_ReLU, .Softmax, .Linear:
		return 0
	}

	return 0
}

predicted_class_for_brain :: proc(brain: ^Brain) -> int {
	last_layer := &brain.layers[len(brain.layers) - 1]
	if brain.config.output_activation == .Softmax {
		return argmax_neurons(last_layer)
	}
	if len(last_layer.neurons) == 0 {
		return -1
	}
	return classify_output(brain.config.output_activation, last_layer.neurons[0].actv)
}

expected_class_for_label :: proc(config: Brain_Config, label: []f32) -> int {
	if config.output_activation == .Softmax {
		return argmax_slice(label)
	}
	if len(label) == 0 {
		return -1
	}
	return classify_output(config.output_activation, label[0])
}

argmax_neurons :: proc(layer: ^Layer) -> int {
	if len(layer.neurons) == 0 {
		return -1
	}

	best_index := 0
	best_value := layer.neurons[0].actv
	for i := 1; i < len(layer.neurons); i += 1 {
		if layer.neurons[i].actv > best_value {
			best_value = layer.neurons[i].actv
			best_index = i
		}
	}

	return best_index
}

argmax_slice :: proc(values: []f32) -> int {
	if len(values) == 0 {
		return -1
	}

	best_index := 0
	best_value := values[0]
	for i := 1; i < len(values); i += 1 {
		if values[i] > best_value {
			best_value = values[i]
			best_index = i
		}
	}

	return best_index
}
apply_softmax :: proc(layer: ^Layer) {
	assert(len(layer.neurons) > 0)

	max_z := layer.neurons[0].z
	for i := 1; i < len(layer.neurons); i += 1 {
		if layer.neurons[i].z > max_z {
			max_z = layer.neurons[i].z
		}
	}

	sum_exp := f32(0)
	for i := 0; i < len(layer.neurons); i += 1 {
		layer.neurons[i].actv = f32(math.exp(f64(layer.neurons[i].z - max_z)))
		sum_exp += layer.neurons[i].actv
	}

	for i := 0; i < len(layer.neurons); i += 1 {
		layer.neurons[i].actv /= sum_exp
	}
}

abs_f32 :: proc(x: f32) -> f32 {
	if x < 0 {
		return -x
	}
	return x
}

// Clamps a probability value to a small range around 0 and 1 to prevent numerical instability when calculating logarithms in the binary cross-entropy loss function.
clamp_probability :: proc(x: f32) -> f32 {
	if x < 1e-6 {
		return 1e-6
	}
	if x > 1 - 1e-6 {
		return 1 - 1e-6
	}
	return x
}
