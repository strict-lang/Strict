// WGSL runtime kernels, run on WebGPU by strict_webgpu_kernel (WebRuntime.js) for strict_gpu_shaded of kind 2: a list
// value of the kernel is a part of numbers at places[2 * slot] with places[2 * slot + 1] numbers, a whole number value
// is places[2 * slot + 1], places[2 * slots] is the count of outputs, one invocation per output if not said otherwise,
// work.x is the first workgroup of a submit of strictKernelGroups, work.y the count of parts an inner sum is split in.
// Lists are read with n_item: bits 28 to 30 of a place say which binding holds the list (0 numbers, 1 to 4 a kept result
// bound as it is), the top bit that it holds bfloat16 halves (two in each 32 bits, strict_packed); every shader starts
// with it (strictWebGpuPipeline), also the generated ones.
Module['strictItem'] = `
@group(0) @binding(5) var<storage, read> lists1: array<f32>;
@group(0) @binding(6) var<storage, read> lists2: array<f32>;
@group(0) @binding(7) var<storage, read> lists3: array<f32>;
@group(0) @binding(8) var<storage, read> lists4: array<f32>;
fn n_number(place: u32, at: u32) -> f32 {
	let index = (place & 0x0fffffffu) + at;
	switch ((place >> 28u) & 7u) {
		case 1u: { return lists1[index]; }
		case 2u: { return lists2[index]; }
		case 3u: { return lists3[index]; }
		case 4u: { return lists4[index]; }
		default: { return numbers[index]; }
	}
}
fn n_item(place: u32, at: u32) -> f32 {
	if (place < 0x80000000u) { return n_number(place, at); }
	let pair = bitcast<u32>(n_number(place & 0x7fffffffu, at / 2u));
	return bitcast<f32>(select(pair << 16u, pair & 0xffff0000u, (at & 1u) == 1u));
}
`;
const strictKernelBindings = `
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> numbers: array<f32>;
@group(0) @binding(2) var<storage, read> places: array<u32>;
@group(0) @binding(3) var<storage, read_write> failed: array<u32>;
@group(0) @binding(4) var<uniform> work: vec2<u32>;
`;
// the 4 by 4 sums plus the products of ins and outs: fma, as Dawn's DXC compiles a * b + c strictly as two operations
const strictSummed = `
fn summed(sums: mat4x4<f32>, ins: vec4<f32>, outs: vec4<f32>) -> mat4x4<f32> {
	return mat4x4<f32>(fma(ins, vec4<f32>(outs.x), sums[0]), fma(ins, vec4<f32>(outs.y), sums[1]),
		fma(ins, vec4<f32>(outs.z), sums[2]), fma(ins, vec4<f32>(outs.w), sums[3]));
}
`;
// Convolved.Summed: values, affine, weights, height, width, channels, outputs, swish, factor (strict_gpu_convolution),
// a workgroup of 256 invocations per tile of 128 pixels of one parity and 128 outputs (8 by 8 sums each), the taps 16
// at a time in workgroup memory; values normed by strict_normed first come with swish 2
(Module['strictShaders'] ||= {})['strict_convolution'] = strictKernelBindings + strictSummed + `
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
	let tile = work.x + group.x;
	if (tile >= factor * factor * pixelTiles * outputTiles) { return; }
	let parity = tile / (pixelTiles * outputTiles);
	let pixelStart = tile / outputTiles % pixelTiles * 128u;
	let outputStart = tile % outputTiles * 128u;
	let parityRow = parity / factor;
	let parityColumn = parity % factor;
	let weights = parity * outputs * taps;
	let ownPixel = local / 16u;
	let ownOutput = local % 16u;
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
				value = n_item(places[0], u32(sourceRow * i32(inputWidth) + sourceColumn) * channels + channel);
				if (!isNormed) {
					let normed = value * n_item(places[2], channel) + n_item(places[2], channels + channel);
					value = select(normed, normed / (1.0 + exp(-normed)), swish);
				}
			}
			tapped[slot / 4u][slot % 4u] = value;
			let produced = outputStart + item;
			var weight = 0.0;
			if (tap < taps && produced < outputs) { weight = n_item(places[4], weights + produced * taps + tap); }
			weighted[slot / 4u][slot % 4u] = weight;
		}
		workgroupBarrier();
		for (var at = 0u; at < 16u; at++) {
			let ins = tapped[at * 32u + ownPixel * 2u];
			let moreIns = tapped[at * 32u + ownPixel * 2u + 1u];
			let outs = weighted[at * 32u + ownOutput * 2u];
			let moreOuts = weighted[at * 32u + ownOutput * 2u + 1u];
			sums = summed(sums, ins, outs);
			moreOutputSums = summed(moreOutputSums, ins, moreOuts);
			morePixelSums = summed(morePixelSums, moreIns, outs);
			moreSums = summed(moreSums, moreIns, moreOuts);
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
// the tiles (one workgroup each) and how many of them go into one submit: about 4 billion multiplications, the
// convolution at least 128 tiles (56 tiles of 4608 taps, fewer than the GPU has multiprocessors, made 360x640 of 512
// channels take 2 s instead of 0.19; 16 billion for all made 720x1280 decodes slower and noisy)
(Module['strictKernelGroups'] ||= {})['strict_convolution'] = (places) => {
	const factor = places[17], taps = (4 - factor) * (4 - factor) * places[11];
	const pixels = places[7] / factor * (places[9] / factor);
	return [factor * factor * Math.ceil(pixels / 128) * Math.ceil(places[13] / 128), Math.max(Math.floor(2 ** 32 / (16384 * taps)), 128)];
};
// Products.Multiplied: left, right, batches, rows, inner, columns, then the start, batch, row and inner strides of left
// and the start, batch, inner and column strides of right (strict_gpu_products): a workgroup of 256 invocations per
// tile of 128 rows and 128 columns of one batch (8 by 8 sums each), the inner numbers 16 at a time in workgroup memory;
// with work.y splits each tile sums a part of the inner numbers into its own copy of the outputs (strict_splits)
Module['strictShaders']['strict_products'] = strictKernelBindings + strictSummed + `
var<workgroup> lefts: array<vec4<f32>, 512>;
var<workgroup> rights: array<vec4<f32>, 512>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
	let rows = places[7];
	let inner = places[9];
	let columns = places[11];
	let rowTiles = (rows + 127u) / 128u;
	let columnTiles = (columns + 127u) / 128u;
	let tiles = places[5] * rowTiles * columnTiles;
	if (work.x + group.x >= tiles * work.y) { return; }
	let split = (work.x + group.x) / tiles;
	let tile = (work.x + group.x) % tiles;
	let part = (inner + work.y * 16u - 1u) / (work.y * 16u) * 16u;
	let first = split * part;
	let last = min(inner, first + part);
	let batch = tile / (rowTiles * columnTiles);
	let rowStart = tile / columnTiles % rowTiles * 128u;
	let columnStart = tile % columnTiles * 128u;
	let leftBase = places[13] + batch * places[15];
	let rightBase = places[21] + batch * places[23];
	let ownRow = local / 16u;
	let ownColumn = local % 16u;
	let ownInner = local % 16u;
	let firstItem = local / 16u;
	var sums = mat4x4<f32>();
	var moreColumnSums = mat4x4<f32>();
	var moreRowSums = mat4x4<f32>();
	var moreSums = mat4x4<f32>();
	for (var start = first; start < last; start += 16u) {
		let step = start + ownInner;
		for (var load = 0u; load < 8u; load++) {
			let item = firstItem + load * 16u;
			let slot = ownInner * 128u + item;
			let row = rowStart + item;
			var left = 0.0;
			if (step < last && row < rows) { left = n_item(places[0], leftBase + row * places[17] + step * places[19]); }
			lefts[slot / 4u][slot % 4u] = left;
			let column = columnStart + item;
			var right = 0.0;
			if (step < last && column < columns) { right = n_item(places[2], rightBase + step * places[25] + column * places[27]); }
			rights[slot / 4u][slot % 4u] = right;
		}
		workgroupBarrier();
		for (var at = 0u; at < 16u; at++) {
			let ins = lefts[at * 32u + ownRow * 2u];
			let moreIns = lefts[at * 32u + ownRow * 2u + 1u];
			let outs = rights[at * 32u + ownColumn * 2u];
			let moreOuts = rights[at * 32u + ownColumn * 2u + 1u];
			sums = summed(sums, ins, outs);
			moreColumnSums = summed(moreColumnSums, ins, moreOuts);
			moreRowSums = summed(moreRowSums, moreIns, outs);
			moreSums = summed(moreSums, moreIns, moreOuts);
		}
		workgroupBarrier();
	}
	var blocks = array<mat4x4<f32>, 4>(sums, moreColumnSums, moreRowSums, moreSums);
	for (var row = 0u; row < 8u; row++) {
		let produced = rowStart + ownRow * 8u + row;
		if (produced >= rows) { continue; }
		let placed = ((split * places[5] + batch) * rows + produced) * columns;
		for (var column = 0u; column < 8u; column++) {
			let placedColumn = columnStart + ownColumn * 8u + column;
			if (placedColumn < columns) { output[placed + placedColumn] = blocks[row / 4u * 2u + column / 4u][column % 4u][row % 4u]; }
		}
	}
}`;
// fewer than 256 tiles split inner sums of at least 1024 numbers: Flux attention's 1165x14400 by 14400x512 had 40 tiles
// for the 68 multiprocessors of an RTX 3080
Module['strictKernelGroups']['strict_products'] = (places) => {
	const tiles = places[5] * Math.ceil(places[7] / 128) * Math.ceil(places[11] / 128);
	const splits = Math.max(1, Math.min(Math.floor(256 / tiles), Math.floor(places[9] / 1024)));
	return [tiles * splits, Math.floor(2 ** 32 / (16384 * Math.ceil(places[9] / splits))) || 1, splits];
};
// the sums of the work.y parts strict_products wrote one after another into numbers
Module['strictShaders']['strict_splits'] = strictKernelBindings + `
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>) {
	let count = arrayLength(&output);
	let at = (work.x + group.y * groups.x + group.x) * 64u + local;
	if (at >= count) { return; }
	var sum = 0.0;
	for (var split = 0u; split < work.y; split++) { sum += numbers[split * count + at]; }
	output[at] = sum;
}`;
// Batch.Sums tokens as 8 bit codes (strict_gpu_tokens_web): input, width; a workgroup of 64 invocations per token writes
// its codes 4 to a word (127 / its largest absolute number, rounded to the nearest even) and after all words the scales
Module['strictShaders']['strict_quantized'] = strictKernelBindings + `
var<workgroup> largests: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>) {
	let width = places[3];
	let tokens = places[1] / width;
	let token = work.x + group.x + group.y * groups.x;
	if (token >= tokens) { return; }
	let first = token * width;
	var largest = 0.0;
	for (var column = local; column < width; column += 64u) { largest = max(largest, abs(n_item(places[0], first + column))); }
	largests[local] = largest;
	workgroupBarrier();
	for (var step = 32u; step > 0u; step /= 2u) {
		if (local < step) { largests[local] = max(largests[local], largests[local + step]); }
		workgroupBarrier();
	}
	let most = largests[0];
	let inverse = select(127.0 / most, 0.0, most == 0.0);
	let words = width / 4u;
	for (var word = local; word < words; word += 64u) {
		let at = first + word * 4u;
		let numbers = vec4<f32>(n_item(places[0], at), n_item(places[0], at + 1u), n_item(places[0], at + 2u),
			n_item(places[0], at + 3u));
		output[token * words + word] = bitcast<f32>(pack4xI8(vec4<i32>(round(numbers * inverse))));
	}
	if (local == 0u) { output[tokens * words + token] = most / 127.0; }
}`;
Module['strictKernelGroups']['strict_quantized'] = (places) => {
	const tokens = places[1] / places[3];
	return [tokens, tokens];
};
// Batch.Sums of 8 bit rows (strict_gpu_tokens_web): rows (codes, then a scale per 32 columns, stride words a row), codes
// and scales of the tokens (strict_quantized), biases, columns, rows, stride, tokens: a workgroup of 256 invocations per
// tile of 64 rows and 64 tokens (4 by 4 each), the codes of 32 columns of both in workgroup memory at a time, dot
// products of 4 codes summed as whole numbers per 32 columns and scaled by their row
Module['strictShaders']['strict_eights'] = strictKernelBindings + `
var<workgroup> rowCodes: array<u32, 512>;
var<workgroup> tokenCodes: array<u32, 512>;
var<workgroup> rowScales: array<f32, 64>;
fn dotted(row: u32, token: u32) -> vec4<f32> {
	var wholes = vec4<i32>();
	for (var word = 0u; word < 8u; word++) {
		let codes = rowCodes[row * 8u + word];
		wholes += vec4<i32>(dot4I8Packed(codes, tokenCodes[token * 8u + word]), dot4I8Packed(codes,
			tokenCodes[(token + 1u) * 8u + word]), dot4I8Packed(codes, tokenCodes[(token + 2u) * 8u + word]),
			dot4I8Packed(codes, tokenCodes[(token + 3u) * 8u + word]));
	}
	return vec4<f32>(wholes) * rowScales[row];
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>) {
	let columns = places[7];
	let rows = places[9];
	let stride = places[11];
	let tokens = places[13];
	let words = columns / 4u;
	let rowTiles = (rows + 63u) / 64u;
	let tile = work.x + group.x + group.y * groups.x;
	if (tile >= rowTiles * ((tokens + 63u) / 64u)) { return; }
	let rowStart = tile % rowTiles * 64u;
	let tokenStart = tile / rowTiles * 64u;
	let ownRow = local % 16u * 4u;
	let ownToken = local / 16u * 4u;
	var first = vec4<f32>();
	var second = vec4<f32>();
	var third = vec4<f32>();
	var fourth = vec4<f32>();
	for (var block = 0u; block < columns / 32u; block++) {
		for (var at = local; at < 512u; at += 256u) {
			let row = rowStart + at / 8u;
			let token = tokenStart + at / 8u;
			rowCodes[at] = select(0u, bitcast<u32>(n_item(places[0], row * stride + block * 8u + at % 8u)), row < rows);
			tokenCodes[at] = select(0u, bitcast<u32>(n_item(places[2], token * words + block * 8u + at % 8u)), token < tokens);
		}
		if (local < 64u) {
			rowScales[local] = select(0.0, n_item(places[0], (rowStart + local) * stride + words + block), rowStart + local < rows);
		}
		workgroupBarrier();
		first += dotted(ownRow, ownToken);
		second += dotted(ownRow + 1u, ownToken);
		third += dotted(ownRow + 2u, ownToken);
		fourth += dotted(ownRow + 3u, ownToken);
		workgroupBarrier();
	}
	let biases = places[5];
	let sums = mat4x4<f32>(first, second, third, fourth);
	for (var token = 0u; token < 4u; token++) {
		let placedToken = tokenStart + ownToken + token;
		if (placedToken >= tokens) { break; }
		let scale = n_item(places[2], tokens * words + placedToken);
		for (var row = 0u; row < 4u; row++) {
			let placedRow = rowStart + ownRow + row;
			let at = placedToken * rows + placedRow;
			let bias = select(0.0, n_item(places[4], at % max(biases, 1u)), biases > 0u);
			if (placedRow < rows) { output[at] = sums[row][token] * scale + bias; }
		}
	}
}`;
// the tiles of strict_eights, about 4 billion multiplications in a submit
Module['strictKernelGroups']['strict_eights'] = (places) => {
	const tiles = Math.ceil(places[9] / 64) * Math.ceil(places[13] / 64);
	return [tiles, Math.max(Math.floor(2 ** 32 / (4096 * places[7])), 1)];
};
// Attention.Merged (strict_gpu_attention_web): queries, keys, values, heads, groups, size (up to 128), sequences,
// causal, value offset, value stride; a workgroup of 64 invocations per 16 tokens of one head and sequence scores 64
// positions at a time (one each), keeps the largest score and the sum of each row (online softmax) and adds the
// weighted values of 2 numbers of all 16 rows per invocation, no scores are stored
Module['strictShaders']['strict_attention'] = strictKernelBindings + `
var<workgroup> asked: array<f32, 2048>;
var<workgroup> scored: array<f32, 1024>;
var<workgroup> largests: array<f32, 16>;
var<workgroup> rescales: array<f32, 16>;
var<workgroup> totals: array<f32, 16>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>) {
	let heads = places[7];
	let keyGroups = places[9];
	let size = places[11];
	let sequences = places[13];
	let causal = places[15] == 1u;
	let valueOffset = places[17];
	let valueStride = select(places[19], sequences * keyGroups * size, places[19] == 0u);
	let tokens = places[1] / heads / size / sequences;
	let positions = places[3] / keyGroups / size / sequences;
	let tokenBlocks = (tokens + 15u) / 16u;
	let block = work.x + group.x + group.y * groups.x;
	if (block >= sequences * tokenBlocks * heads) { return; }
	let head = block % heads;
	let sequence = block / heads / tokenBlocks;
	let firstToken = block / heads % tokenBlocks * 16u;
	let keyGroup = sequence * keyGroups + head * keyGroups / heads;
	for (var at = local; at < 16u * size; at += 64u) {
		let token = firstToken + at / size;
		var query = 0.0;
		if (token < tokens) { query = n_item(places[0], ((sequence * tokens + token) * heads + head) * size + at % size); }
		asked[at] = query / sqrt(f32(size));
	}
	if (local < 16u) {
		largests[local] = -3.0e38;
		totals[local] = 0.0;
	}
	let lastToken = min(firstToken + 15u, tokens - 1u);
	let visible = select(positions, positions - tokens + lastToken + 1u, causal);
	var sums: array<f32, 32>;
	for (var start = 0u; start < visible; start += 64u) {
		workgroupBarrier();
		let position = start + local;
		var scores: array<f32, 16>;
		if (position < visible) {
			let keyStart = (position * sequences * keyGroups + keyGroup) * size;
			for (var at = 0u; at < size; at++) {
				let key = n_item(places[2], keyStart + at);
				for (var row = 0u; row < 16u; row++) { scores[row] = fma(asked[row * size + at], key, scores[row]); }
			}
		}
		for (var row = 0u; row < 16u; row++) {
			let token = firstToken + row;
			let last = select(positions - 1u, positions - tokens + token, causal);
			scored[row * 64u + local] = select(-3.0e38, scores[row], position < visible && position <= last);
		}
		workgroupBarrier();
		if (local < 16u) {
			var largest = largests[local];
			for (var at = 0u; at < 64u; at++) { largest = max(largest, scored[local * 64u + at]); }
			var total = 0.0;
			for (var at = 0u; at < 64u; at++) {
				let share = select(exp(scored[local * 64u + at] - largest), 0.0, scored[local * 64u + at] < -1.0e38);
				scored[local * 64u + at] = share;
				total += share;
			}
			let rescale = exp(largests[local] - largest);
			rescales[local] = rescale;
			totals[local] = totals[local] * rescale + total;
			largests[local] = largest;
		}
		workgroupBarrier();
		for (var row = 0u; row < 16u; row++) {
			sums[row * 2u] *= rescales[row];
			sums[row * 2u + 1u] *= rescales[row];
		}
		let count = min(64u, visible - start);
		for (var at = 0u; at < count; at++) {
			let valueStart = valueOffset + keyGroup * size + (start + at) * valueStride;
			let first = select(0.0, n_item(places[4], valueStart + local), local < size);
			let second = select(0.0, n_item(places[4], valueStart + local + 64u), local + 64u < size);
			for (var row = 0u; row < 16u; row++) {
				let share = scored[row * 64u + at];
				sums[row * 2u] = fma(share, first, sums[row * 2u]);
				sums[row * 2u + 1u] = fma(share, second, sums[row * 2u + 1u]);
			}
		}
	}
	workgroupBarrier();
	for (var row = 0u; row < 16u; row++) {
		let token = firstToken + row;
		if (token >= tokens) { break; }
		let placed = ((sequence * tokens + token) * heads + head) * size;
		let total = totals[row];
		if (local < size) { output[placed + local] = sums[row * 2u] / total; }
		if (local + 64u < size) { output[placed + local + 64u] = sums[row * 2u + 1u] / total; }
	}
}`;
// the blocks of 16 tokens of strict_attention, about 4 billion multiplications in a submit
Module['strictKernelGroups']['strict_attention'] = (places) => {
	const tokens = places[1] / places[7] / places[11] / places[13], blocks = places[13] * Math.ceil(tokens / 16) * places[7];
	const positions = places[3] / places[9] / places[11] / places[13];
	return [blocks, Math.max(Math.floor(2 ** 32 / (32 * positions * places[11])), 1)];
};
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
		let first = row * width;
		var largest = -0x1.fffffep+127f;
		for (var column = local; column < width; column += 64u) { largest = max(largest, n_item(places[0], first + column)); }
		largest = reduced(largest, local, false);
		var total = 0.0;
		for (var column = local; column < width; column += 64u) { total += exp(n_item(places[0], first + column) - largest); }
		total = reduced(total, local, true);
		for (var column = local; column < width; column += 64u) {
			output[row * width + column] = exp(n_item(places[0], first + column) - largest) / total;
		}
	}
}`;
// a kept bfloat16 result (format 17) rounded to the nearest even bfloat16 like the CPU reads it, two in each 32 bits
Module['strictShaders']['strict_packed'] = `
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<storage, read> numbers: array<f32>;
fn rounded(number: f32) -> u32 {
	let bits = bitcast<u32>(number);
	return (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let at = id.x + id.y * groups.x * 64u;
	let count = arrayLength(&numbers);
	if (at * 2u >= count) { return; }
	let second = select(0u, rounded(numbers[min(at * 2u + 1u, count - 1u)]) << 16u, at * 2u + 1u < count);
	output[at] = rounded(numbers[at * 2u]) | second;
}`;
// the values of a convolution normed with affine and swished when swish is 1 once (strict_gpu_convolution), not in
// each of its tiles: values, affine, channels, swish
Module['strictShaders']['strict_normed'] = strictKernelBindings + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let at = id.x + id.y * groups.x * 64u;
	if (at >= places[8]) { return; }
	let channel = at % places[5];
	let normed = n_item(places[0], at) * n_item(places[2], channel) + n_item(places[2], places[5] + channel);
	output[at] = select(normed, normed / (1.0 + exp(-normed)), places[7] == 1u);
}`;
