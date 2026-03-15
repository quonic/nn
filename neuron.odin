package nn

Neuron :: struct {
	actv:        f32,
	out_weights: []f32,
	bias:        f32,
	z:           f32,
	dactv:       f32,
	dw:          []f32,
	dbias:       f32,
	dz:          f32,
}

make_neuron :: proc(num_out_weights: int) -> (neuron: Neuron) {
	neuron.out_weights = make([]f32, num_out_weights)
	neuron.dw = make([]f32, num_out_weights)
	return
}

destroy_neuron :: proc(neuron: ^Neuron) {
	delete(neuron.out_weights)
	delete(neuron.dw)
	neuron^ = {}
}
