// WGSL runtime kernels, run on WebGPU by strict_webgpu_kernel (WebRuntime.js) for strict_gpu_shaded of kind 2: a list
// value of the kernel is a part of numbers at places[2 * slot] with places[2 * slot + 1] numbers, a whole number value
// is places[2 * slot + 1], places[2 * slots] is the count of outputs, one invocation per output if not said otherwise.
const strictKernelBindings = `
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> numbers: array<f32>;
@group(0) @binding(2) var<storage, read> places: array<u32>;
@group(0) @binding(3) var<storage, read_write> failed: array<u32>;
`;
// Convolved.Summed: values, affine, weights, height, width, channels, outputs, swish, factor (strict_gpu_convolution)
(Module['strictShaders'] ||= {})['strict_convolution'] = strictKernelBindings + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let at = id.x + id.y * groups.x * 64u;
	if (at >= places[18]) { return; }
	let width = places[9];
	let channels = places[11];
	let outputs = places[13];
	let swish = places[15] == 1u;
	let factor = places[17];
	let pixel = at / outputs;
	let row = pixel / width;
	let column = pixel % width;
	let side = 4u - factor;
	let parity = select(0u, (row % 2u) * 2u + column % 2u, factor == 2u);
	let first = places[4] + (parity * outputs + at % outputs) * side * side * channels;
	let inputHeight = i32(places[7] / factor);
	let inputWidth = i32(width / factor);
	var sum = 0.0;
	for (var tap = 0u; tap < side * side; tap++) {
		let sourceRow = i32(row / factor + tap / side + row % factor) - 1;
		let sourceColumn = i32(column / factor + tap % side + column % factor) - 1;
		if (sourceRow < 0 || sourceColumn < 0 || sourceRow >= inputHeight || sourceColumn >= inputWidth) { continue; }
		let source = places[0] + u32(sourceRow * inputWidth + sourceColumn) * channels;
		let weight = first + tap * channels;
		for (var channel = 0u; channel < channels; channel++) {
			let normed = numbers[source + channel] * numbers[places[2] + channel] + numbers[places[2] + channels + channel];
			sum += select(normed, normed / (1.0 + exp(-normed)), swish) * numbers[weight + channel];
		}
	}
	output[at] = sum;
}`;
// Products.Multiplied: left, right, batches, rows, inner, columns, then the start, batch, row and inner strides of left
// and the start, batch, inner and column strides of right (strict_gpu_products)
Module['strictShaders']['strict_products'] = strictKernelBindings + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let at = id.x + id.y * groups.x * 64u;
	if (at >= places[28]) { return; }
	let rows = places[7];
	let columns = places[11];
	let batch = at / (rows * columns);
	var left = places[0] + places[13] + batch * places[15] + (at / columns) % rows * places[17];
	var right = places[2] + places[21] + batch * places[23] + at % columns * places[27];
	var sum = 0.0;
	for (var index = 0u; index < places[9]; index++) {
		sum += numbers[left] * numbers[right];
		left += places[19];
		right += places[25];
	}
	output[at] = sum;
}`;
// Softmax.Shares: scores, width, a workgroup of 64 invocations per row (strict_gpu_softmax)
Module['strictShaders']['strict_softmax'] = strictKernelBindings + `
var<workgroup> partials: array<f32, 64>;
fn reduced(own: f32, local: u32, isSum: bool) -> f32 {
	partials[local] = own;
	workgroupBarrier();
	for (var half = 32u; half > 0u; half /= 2u) {
		if (local < half) {
			let other = partials[local + half];
			partials[local] = select(max(partials[local], other), partials[local] + other, isSum);
		}
		workgroupBarrier();
	}
	let result = partials[0];
	workgroupBarrier();
	return result;
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>) {
	let width = places[3];
	let rows = places[4] / width;
	for (var row = group.x + group.y * groups.x; row < rows; row += groups.x * groups.y) {
		let first = places[0] + row * width;
		var largest = -0x1.fffffep+127f;
		for (var column = local; column < width; column += 64u) { largest = max(largest, numbers[first + column]); }
		largest = reduced(largest, local, false);
		var total = 0.0;
		for (var column = local; column < width; column += 64u) { total += exp(numbers[first + column] - largest); }
		total = reduced(total, local, true);
		for (var column = local; column < width; column += 64u) {
			output[row * width + column] = exp(numbers[first + column] - largest) / total;
		}
	}
}`;
