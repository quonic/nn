package nn

Layer :: struct {
	neurons: []Neuron,
}

make_layer :: proc(num_neurons, next_layer_size: int) -> (layer: Layer) {
	layer.neurons = make([]Neuron, num_neurons)
	for i := 0; i < num_neurons; i += 1 {
		layer.neurons[i] = make_neuron(next_layer_size)
	}
	return
}

destroy_layer :: proc(layer: ^Layer) {
	for i := 0; i < len(layer.neurons); i += 1 {
		destroy_neuron(&layer.neurons[i])
	}

	delete(layer.neurons)
	layer^ = {}
}
