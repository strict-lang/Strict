// WGSL runtime kernels, run on WebGPU by strict_webgpu_kernel (WebRuntime.js) for strict_gpu_shaded of kind 2: a list
// value of the kernel is a part of numbers at places[2 * slot] with places[2 * slot + 1] numbers, a whole number value
// is places[2 * slot + 1], places[2 * slots] is the count of outputs, one invocation per output if not said otherwise,
// firstGroup is the first workgroup of a submit of strictKernelGroups.
const strictKernelBindings = `
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> numbers: array<f32>;
@group(0) @binding(2) var<storage, read> places: array<u32>;
@group(0) @binding(3) var<storage, read_write> failed: array<u32>;
@group(0) @binding(4) var<uniform> firstGroup: u32;
`;
// Convolved.Summed: values, affine, weights, height, width, channels, outputs, swish, factor (strict_gpu_convolution),
// a workgroup of 256 invocations per tile of 128 pixels of one parity and 128 outputs (8 by 8 sums each), the taps 16
// at a time in workgroup memory; values normed by strict_normed first come with swish 2
(Module['strictShaders'] ||= {})['strict_convolution'] = strictKernelBindings + `
var<workgroup> tapped: array<vec4<f32>, 512>;
var<workgroup> weighted: array<vec4<f32>, 512>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
	let width = places[9];
	let channels = places[11];
	let outputs = places[13];
	let swish = places[15] == 1u;
	let isNormed = places[15] == 2u;
	let factor = places[17];
	let side = 4u - factor;
	let taps = side * side * channels;
	let inputHeight = places[7] / factor;
	let inputWidth = width / factor;
	let pixels = inputHeight * inputWidth;
	let pixelTiles = (pixels + 127u) / 128u;
	let outputTiles = (outputs + 127u) / 128u;
	let tile = firstGroup + group.x;
	if (tile >= factor * factor * pixelTiles * outputTiles) { return; }
	let parity = tile / (pixelTiles * outputTiles);
	let pixelStart = tile / outputTiles % pixelTiles * 128u;
	let outputStart = tile % outputTiles * 128u;
	let parityRow = parity / factor;
	let parityColumn = parity % factor;
	let weights = places[4] + parity * outputs * taps;
	let ownPixel = local % 16u;
	let ownOutput = local / 16u;
	let ownTap = local % 16u;
	let firstItem = local / 16u;
	var sums = mat4x4<f32>();
	var moreOutputSums = mat4x4<f32>();
	var morePixelSums = mat4x4<f32>();
	var moreSums = mat4x4<f32>();
	for (var start = 0u; start < taps; start += 16u) {
		let tap = start + ownTap;
		let position = tap / channels;
		let channel = tap % channels;
		let tapRow = i32(position / side + parityRow) - 1;
		let tapColumn = i32(position % side + parityColumn) - 1;
		for (var load = 0u; load < 8u; load++) {
			let item = firstItem + load * 16u;
			let slot = ownTap * 128u + item;
			let pixel = pixelStart + item;
			var value = 0.0;
			let sourceRow = i32(pixel / inputWidth) + tapRow;
			let sourceColumn = i32(pixel % inputWidth) + tapColumn;
			let isInside = sourceRow >= 0 && sourceColumn >= 0 && sourceRow < i32(inputHeight) && sourceColumn < i32(inputWidth);
			if (tap < taps && pixel < pixels && isInside) {
				value = numbers[places[0] + u32(sourceRow * i32(inputWidth) + sourceColumn) * channels + channel];
				if (!isNormed) {
					let normed = value * numbers[places[2] + channel] + numbers[places[2] + channels + channel];
					value = select(normed, normed / (1.0 + exp(-normed)), swish);
				}
			}
			tapped[slot / 4u][slot % 4u] = value;
			let produced = outputStart + item;
			var weight = 0.0;
			if (tap < taps && produced < outputs) { weight = numbers[weights + produced * taps + tap]; }
			weighted[slot / 4u][slot % 4u] = weight;
		}
		workgroupBarrier();
		for (var at = 0u; at < 16u; at++) {
			let ins = tapped[at * 32u + ownPixel * 2u];
			let moreIns = tapped[at * 32u + ownPixel * 2u + 1u];
			let outs = weighted[at * 32u + ownOutput * 2u];
			let moreOuts = weighted[at * 32u + ownOutput * 2u + 1u];
			sums += mat4x4<f32>(ins * outs.x, ins * outs.y, ins * outs.z, ins * outs.w);
			moreOutputSums += mat4x4<f32>(ins * moreOuts.x, ins * moreOuts.y, ins * moreOuts.z, ins * moreOuts.w);
			morePixelSums += mat4x4<f32>(moreIns * outs.x, moreIns * outs.y, moreIns * outs.z, moreIns * outs.w);
			moreSums += mat4x4<f32>(moreIns * moreOuts.x, moreIns * moreOuts.y, moreIns * moreOuts.z, moreIns * moreOuts.w);
		}
		workgroupBarrier();
	}
	var blocks = array<mat4x4<f32>, 4>(sums, moreOutputSums, morePixelSums, moreSums);
	for (var row = 0u; row < 8u; row++) {
		let pixel = pixelStart + ownPixel * 8u + row;
		if (pixel >= pixels) { continue; }
		let placed = ((pixel / inputWidth * factor + parityRow) * width + pixel % inputWidth * factor + parityColumn) * outputs;
		for (var column = 0u; column < 8u; column++) {
			let produced = outputStart + ownOutput * 8u + column;
			if (produced < outputs) { output[placed + produced] = blocks[row / 4u * 2u + column / 4u][column % 4u][row % 4u]; }
		}
	}
}`;
// the tiles (one workgroup each) and how many of them go into one submit, about 4 billion multiplications each
Module['strictKernelGroups'] = { strict_convolution: (places) => {
	const factor = places[17], taps = (4 - factor) * (4 - factor) * places[11];
	const pixels = places[7] / factor * (places[9] / factor);
	return [factor * factor * Math.ceil(pixels / 128) * Math.ceil(places[13] / 128), Math.floor(2 ** 32 / (16384 * taps)) || 1];
} };
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
// a kept bfloat16 result (format 17) rounded to the nearest even bfloat16 where it is, like the CPU reads it
Module['strictShaders']['strict_bfloat16'] = strictKernelBindings + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let at = id.x + id.y * groups.x * 64u;
	if (at >= arrayLength(&output)) { return; }
	let bits = bitcast<u32>(output[at]);
	output[at] = bitcast<f32>((bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u);
}`;
// the values of a convolution normed with affine and swished when swish is 1 once (strict_gpu_convolution), not in
// each of its tiles: values, affine, channels, swish
Module['strictShaders']['strict_normed'] = strictKernelBindings + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let at = id.x + id.y * groups.x * 64u;
	if (at >= places[8]) { return; }
	let channel = at % places[5];
	let normed = numbers[places[0] + at] * numbers[places[2] + channel] + numbers[places[2] + places[5] + channel];
	output[at] = select(normed, normed / (1.0 + exp(-normed)), places[7] == 1u);
}`;
