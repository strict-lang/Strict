// WGSL runtime kernels, run on WebGPU by strict_webgpu_kernel (WebRuntime.js) for strict_gpu_shaded of kind 2: a list
// value of the kernel is a part of numbers at places[2 * slot] with places[2 * slot + 1] numbers, a whole number value
// is places[2 * slot + 1], places[2 * slots] is the count of outputs, one invocation per output if not said otherwise,
// work.x is the first workgroup of a submit of strictKernelGroups, work.y the count of parts an inner sum is split in.
// Lists are read with n_item: bits 28 to 30 of a place say which binding holds the list (0 numbers, 1 to 4 a kept result
// bound as it is), the top bit that it holds bfloat16 halves (two in each 32 bits, strict_packed), with bit 27 IEEE 16
// bit halves (Moss's dense rows); every shader starts with it (strictWebGpuPipeline), also the generated ones.
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
	let pair = bitcast<u32>(n_number(place & 0x77ffffffu, at / 2u));
	if ((place & 0x08000000u) != 0u) { return unpack2x16float(pair)[at & 1u]; }
	return bitcast<f32>(select(pair << 16u, pair & 0xffff0000u, (at & 1u) == 1u));
}
fn n_pair(place: u32, at: u32) -> vec2<f32> {
	if (place < 0x80000000u || (at & 1u) == 1u) { return vec2<f32>(n_item(place, at), n_item(place, at + 1u)); }
	let word = bitcast<u32>(n_number(place & 0x77ffffffu, at / 2u));
	if ((place & 0x08000000u) != 0u) { return unpack2x16float(word); }
	return vec2<f32>(bitcast<f32>(word << 16u), bitcast<f32>(word & 0xffff0000u));
}
fn bf16_half(number: f32) -> u32 {
	let bits = bitcast<u32>(number);
	return (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u;
}
fn pack_bf16(first: f32, second: f32) -> f32 {
	return bitcast<f32>(bf16_half(first) | (bf16_half(second) << 16u));
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
// at a time in workgroup memory (padded stride 132); swish is the Activation kind of the normed values (5 swish, 2 relu,
// 6 linear), values normed by strict_normed first come with 255
const strictConvolution = (half = false, packed = false) => {
	const sc = half ? 'f16' : 'f32', zero = half ? '0.0h' : '0.0';
	return (half ? 'enable f16;\n' : '') + strictKernelBindings + `
fn summed_conv(sums: mat4x4<${sc}>, ins: vec4<${sc}>, outs: vec4<${sc}>) -> mat4x4<${sc}> {
	return mat4x4<${sc}>(fma(ins, vec4<${sc}>(outs.x), sums[0]), fma(ins, vec4<${sc}>(outs.y), sums[1]),
		fma(ins, vec4<${sc}>(outs.z), sums[2]), fma(ins, vec4<${sc}>(outs.w), sums[3]));
}
var<workgroup> tapped: array<${sc}, 2112>;
var<workgroup> weighted: array<${sc}, 2112>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
	let width = places[9];
	let channels = places[11];
	let outputs = places[13];
	let activation = places[15];
	let isNormed = activation == 255u;
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
	let weightsBase = parity * outputs * taps;
	let ownPixel = local / 16u;
	let ownOutput = local % 16u;
	let kQuad = local % 4u;
	let firstItem = local / 4u;
	let valPlace = places[0];
	let wPlace = places[4];
	let valBase = valPlace & 0x0fffffffu;
	let wBase = wPlace & 0x0fffffffu;
	let fastBf16 = isNormed && (valPlace & 0xf8000000u) == 0x90000000u && (wPlace & 0xf8000000u) == 0xb0000000u && (channels & 3u) == 0u;
	var pCoords: array<u32, 2>;
	for (var load = 0u; load < 2u; load++) {
		let px = pixelStart + firstItem + load * 64u;
		pCoords[load] = select(0xffffffffu, ((px / inputWidth) << 16u) | (px % inputWidth), px < pixels);
	}
	var sums = mat4x4<${sc}>();
	var moreOutputSums = mat4x4<${sc}>();
	var morePixelSums = mat4x4<${sc}>();
	var moreSums = mat4x4<${sc}>();
	for (var start = 0u; start < taps; start += 16u) {
		let tap0 = start + kQuad * 4u;
		if (fastBf16) {
			let pos0 = tap0 / channels;
			let ch0 = tap0 % channels;
			let tapRow0 = i32(pos0 / side + parityRow) - 1;
			let tapCol0 = i32(pos0 % side + parityColumn) - 1;
			let tBase = kQuad * 4u;
			for (var load = 0u; load < 2u; load++) {
				let item = firstItem + load * 64u;
				let pc = pCoords[load];
				var u0 = 0u;
				var u1 = 0u;
				let sr = i32(pc >> 16u) + tapRow0;
				let sc0 = i32(pc & 0xffffu) + tapCol0;
				if (tap0 < taps && sr >= 0 && sc0 >= 0 && sr < i32(inputHeight) && sc0 < i32(inputWidth)) {
					let idx = ((u32(sr) * inputWidth + u32(sc0)) * channels + ch0) / 2u + valBase;
					u0 = bitcast<u32>(lists1[idx]);
					u1 = bitcast<u32>(lists1[idx + 1u]);
				}
				tapped[(tBase + 0u) * 132u + item] = ${sc}(bitcast<f32>(u0 << 16u));
				tapped[(tBase + 1u) * 132u + item] = ${sc}(bitcast<f32>(u0 & 0xffff0000u));
				tapped[(tBase + 2u) * 132u + item] = ${sc}(bitcast<f32>(u1 << 16u));
				tapped[(tBase + 3u) * 132u + item] = ${sc}(bitcast<f32>(u1 & 0xffff0000u));
				let produced = outputStart + item;
				var wu0 = 0u;
				var wu1 = 0u;
				if (tap0 < taps && produced < outputs) {
					let widx = (weightsBase + produced * taps + tap0) / 2u + wBase;
					wu0 = bitcast<u32>(lists3[widx]);
					wu1 = bitcast<u32>(lists3[widx + 1u]);
				}
				weighted[(tBase + 0u) * 132u + item] = ${sc}(bitcast<f32>(wu0 << 16u));
				weighted[(tBase + 1u) * 132u + item] = ${sc}(bitcast<f32>(wu0 & 0xffff0000u));
				weighted[(tBase + 2u) * 132u + item] = ${sc}(bitcast<f32>(wu1 << 16u));
				weighted[(tBase + 3u) * 132u + item] = ${sc}(bitcast<f32>(wu1 & 0xffff0000u));
			}
		} else {
			for (var sub = 0u; sub < 4u; sub++) {
				let tap = tap0 + sub;
				let position = tap / channels;
				let channel = tap % channels;
				let tapRow = i32(position / side + parityRow) - 1;
				let tapColumn = i32(position % side + parityColumn) - 1;
				for (var load = 0u; load < 2u; load++) {
					let item = firstItem + load * 64u;
					let pc = pCoords[load];
					var value = ${zero};
					let sourceRow = i32(pc >> 16u) + tapRow;
					let sourceColumn = i32(pc & 0xffffu) + tapColumn;
					if (tap < taps && sourceRow >= 0 && sourceColumn >= 0 && sourceRow < i32(inputHeight) && sourceColumn < i32(inputWidth)) {
						var vf = n_item(valPlace, u32(sourceRow * i32(inputWidth) + sourceColumn) * channels + channel);
						if (!isNormed) {
							let normed = vf * n_item(places[2], channel) + n_item(places[2], channels + channel);
							vf = select(select(normed, max(normed, 0.0), activation == 2u), normed / (1.0 + exp(-normed)), activation == 5u);
						}
						value = ${sc}(vf);
					}
					tapped[(kQuad * 4u + sub) * 132u + item] = value;
					let produced = outputStart + item;
					var weight = ${zero};
					if (tap < taps && produced < outputs) { weight = ${sc}(n_item(wPlace, weightsBase + produced * taps + tap)); }
					weighted[(kQuad * 4u + sub) * 132u + item] = weight;
				}
			}
		}
		workgroupBarrier();
		for (var at = 0u; at < 16u; at++) {
			let pBase = at * 132u + ownPixel * 8u;
			let oBase = at * 132u + ownOutput * 8u;
			let ins = vec4<${sc}>(tapped[pBase], tapped[pBase + 1u], tapped[pBase + 2u], tapped[pBase + 3u]);
			let moreIns = vec4<${sc}>(tapped[pBase + 4u], tapped[pBase + 5u], tapped[pBase + 6u], tapped[pBase + 7u]);
			let outs = vec4<${sc}>(weighted[oBase], weighted[oBase + 1u], weighted[oBase + 2u], weighted[oBase + 3u]);
			let moreOuts = vec4<${sc}>(weighted[oBase + 4u], weighted[oBase + 5u], weighted[oBase + 6u], weighted[oBase + 7u]);
			sums = summed_conv(sums, ins, outs);
			moreOutputSums = summed_conv(moreOutputSums, ins, moreOuts);
			morePixelSums = summed_conv(morePixelSums, moreIns, outs);
			moreSums = summed_conv(moreSums, moreIns, moreOuts);
		}
		workgroupBarrier();
	}
	var blocks = array<mat4x4<${sc}>, 4>(sums, moreOutputSums, morePixelSums, moreSums);
	for (var row = 0u; row < 8u; row++) {
		let pixel = pixelStart + ownPixel * 8u + row;
		if (pixel >= pixels) { continue; }
		let placed = ((pixel / inputWidth * factor + parityRow) * width + pixel % inputWidth * factor + parityColumn) * outputs;
${packed ? `		for (var column = 0u; column < 8u; column += 2u) {
			let produced = outputStart + ownOutput * 8u + column;
			if (produced < outputs) {
				let b = row / 4u * 2u + column / 4u;
				output[(placed + produced) / 2u] = pack_bf16(f32(blocks[b][column % 4u][row % 4u]), f32(blocks[b][(column + 1u) % 4u][row % 4u]));
			}
		}` : `		for (var column = 0u; column < 8u; column++) {
			let produced = outputStart + ownOutput * 8u + column;
			if (produced < outputs) { output[placed + produced] = f32(blocks[row / 4u * 2u + column / 4u][column % 4u][row % 4u]); }
		}`}
	}
}`;
};
(Module['strictShaders'] ||= {})['strict_convolution'] = strictConvolution(false, false);
Module['strictShaders']['strict_convolution_p17'] = strictConvolution(false, true);
Module['strictShaders']['strict_convolution_f16'] = strictConvolution(true, false);
Module['strictShaders']['strict_convolution_f16_p17'] = strictConvolution(true, true);
(Module['strictHalf'] ||= {})['strict_convolution'] = 'strict_convolution_f16';
(Module['strictPacked'] ||= {})['strict_convolution'] = 'strict_convolution_p17';
Module['strictPacked']['strict_convolution_f16'] = 'strict_convolution_f16_p17';
(Module['strictCanPack'] ||= {})['strict_convolution'] = (places) => (places[13] & 1) === 0;
Module['strictCanPack']['strict_convolution_f16'] = Module['strictCanPack']['strict_convolution'];
// the tiles (one workgroup each) and how many of them go into one submit: about 4 billion multiplications, the
// convolution at least 128 tiles (56 tiles of 4608 taps, fewer than the GPU has multiprocessors, made 360x640 of 512
// channels take 2 s instead of 0.19; 16 billion for all made 720x1280 decodes slower and noisy)
(Module['strictKernelGroups'] ||= {})['strict_convolution'] = (places) => {
	const factor = places[17], taps = (4 - factor) * (4 - factor) * places[11];
	const pixels = places[7] / factor * (places[9] / factor);
	return [factor * factor * Math.ceil(pixels / 128) * Math.ceil(places[13] / 128), Math.max(Math.floor(2 ** 32 / (16384 * taps)), 128)];
};
Module['strictKernelGroups']['strict_convolution_p17'] = Module['strictKernelGroups']['strict_convolution'];
Module['strictKernelGroups']['strict_convolution_f16'] = Module['strictKernelGroups']['strict_convolution'];
Module['strictKernelGroups']['strict_convolution_f16_p17'] = Module['strictKernelGroups']['strict_convolution'];
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
const strictQuantized = (subgroups = false) => (subgroups ? 'enable subgroups;\n' : '') + strictKernelBindings + `
var<workgroup> largests: array<f32, ${subgroups ? 16 : 64}>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>${subgroups ? ', @builtin(subgroup_invocation_id) sgLane: u32, @builtin(subgroup_size) sgSize: u32' : ''}) {
	let width = places[3];
	let tokens = places[1] / width;
	let token = work.x + group.x + group.y * groups.x;
	if (token >= tokens) { return; }
	let first = token * width;
	var largest = 0.0;
	for (var column = local; column < width; column += 64u) { largest = max(largest, abs(n_item(places[0], first + column))); }
${subgroups ? `	let sgMax = subgroupMax(largest);
	if (sgLane == 0u) { largests[local / sgSize] = sgMax; }
	workgroupBarrier();
	var most = 0.0;
	let sgCount = (64u + sgSize - 1u) / sgSize;
	for (var i = 0u; i < sgCount; i++) { most = max(most, largests[i]); }` : `	largests[local] = largest;
	workgroupBarrier();
	for (var step = 32u; step > 0u; step /= 2u) {
		if (local < step) { largests[local] = max(largests[local], largests[local + step]); }
		workgroupBarrier();
	}
	let most = largests[0];`}
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
Module['strictShaders']['strict_quantized'] = strictQuantized(false);
Module['strictShaders']['strict_quantized_sg'] = strictQuantized(true);
(Module['strictSubgroups'] ||= {})['strict_quantized'] = 'strict_quantized_sg';
Module['strictKernelGroups']['strict_quantized'] = (places) => {
	const tokens = places[1] / places[3];
	return [tokens, tokens];
};
Module['strictKernelGroups']['strict_quantized_sg'] = Module['strictKernelGroups']['strict_quantized'];
// Batch.Sums of 8 bit rows (strict_gpu_tokens_web): rows (codes, then a scale per 32 columns, stride words a row), codes
// and scales of the tokens (strict_quantized), biases, columns, rows, stride, tokens, nibbles (codes as 4 bit halves of
// bytes, unpacked to 8 bit codes as they are staged): a workgroup of 256 invocations per tile of 64 rows and 64 tokens
// (4 neighboring rows and 4 neighboring tokens of it each, read as one vec4 each per word), the codes of 32 columns of
// both in workgroup memory at a time (word by word), dot products of 4 codes summed as whole numbers per 32 columns and
// scaled by their row (rows of only their codes, read from a URL, have one scale each after the codes of all rows)
const strictEightsCodes = (rowFn, tokenFn) => `		if (nibbles) {
			let r = local / 4u;
			let row = rowStart + r;
			let wPair = local % 4u;
			var packed = 0u;
			if (row < rows) { packed = bitcast<u32>(${rowFn('row * stride + block * 4u + wPair')}); }
			rowCodes[(wPair * 2u) * 16u + r / 4u][r % 4u] = unpacked(packed & 0xffffu);
			rowCodes[(wPair * 2u + 1u) * 16u + r / 4u][r % 4u] = unpacked(packed >> 16u);
		} else {
			for (var at = local; at < 512u; at += 256u) {
				let r = at / 8u;
				let row = rowStart + r;
				var code = 0u;
				if (row < rows) { code = bitcast<u32>(${rowFn('row * stride + block * 8u + at % 8u')}); }
				rowCodes[at % 8u * 16u + r / 4u][r % 4u] = code;
			}
		}
		for (var at = local; at < 512u; at += 256u) {
			let t = at / 8u;
			let token = tokenStart + t;
			var tcode = 0u;
			if (token < tokens) { tcode = bitcast<u32>(${tokenFn('token * words + block * 8u + at % 8u')}); }
			tokenCodes[at % 8u * 16u + t / 4u][t % 4u] = tcode;
		}`;
const strictEightsInner = () => `		var firstWholes = vec4<i32>();
		var secondWholes = vec4<i32>();
		var thirdWholes = vec4<i32>();
		var fourthWholes = vec4<i32>();
		for (var word = 0u; word < 8u; word++) {
			let tokenWords = tokenCodes[word * 16u + ownToken];
			let rowWords = rowCodes[word * 16u + ownRow];
			firstWholes += dotted(rowWords.x, tokenWords);
			secondWholes += dotted(rowWords.y, tokenWords);
			thirdWholes += dotted(rowWords.z, tokenWords);
			fourthWholes += dotted(rowWords.w, tokenWords);
		}`;
const strictEights = (packed = false) => strictKernelBindings + `
var<workgroup> rowCodes: array<vec4<u32>, 128>;
var<workgroup> tokenCodes: array<vec4<u32>, 128>;
var<workgroup> rowScales: array<f32, 64>;
fn unpacked(halves: u32) -> u32 {
	let spread = (halves & 0x000fu) | ((halves & 0x00f0u) << 4u) | ((halves & 0x0f00u) << 8u) | ((halves & 0xf000u) << 12u);
	return spread | (((spread & 0x08080808u) >> 3u) * 0xf0u);
}
fn dotted(codes: u32, tokens: vec4<u32>) -> vec4<i32> {
	return vec4<i32>(dot4I8Packed(codes, tokens.x), dot4I8Packed(codes, tokens.y), dot4I8Packed(codes, tokens.z),
		dot4I8Packed(codes, tokens.w));
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>) {
	let columns = places[7];
	let rows = places[9];
	let stride = places[11];
	let tokens = places[13];
	let nibbles = places[15] == 1u;
	let words = columns / 4u;
	let rowWords = select(words, words / 2u, nibbles);
	let packedScale = stride == rowWords;
	let rowTiles = (rows + 63u) / 64u;
	let tokenTiles = (tokens + 63u) / 64u;
	let tile = work.x + group.x + group.y * groups.x;
	if (tile >= rowTiles * tokenTiles) { return; }
	let panel = 4u;
	let strip = tile / (rowTiles * panel);
	let tokenStartTile = strip * panel;
	let stripWidth = min(panel, tokenTiles - tokenStartTile);
	let within = tile % (rowTiles * panel);
	let rowStart = within / stripWidth * 64u;
	let tokenStart = (tokenStartTile + within % stripWidth) * 64u;
	let ownRow = local % 16u;
	let ownToken = local / 16u;
	let isBound = (places[0] & 0xf8000000u) == 0x10000000u && (places[2] & 0xf8000000u) == 0x20000000u;
	let rowBase = places[0] & 0x0fffffffu;
	let tokenBase = places[2] & 0x0fffffffu;
	let blocks = columns / 32u;
	var sums: mat4x4<f32>;
	if (packedScale) {
		var i0 = vec4<i32>();
		var i1 = vec4<i32>();
		var i2 = vec4<i32>();
		var i3 = vec4<i32>();
		for (var block = 0u; block < blocks; block++) {
			if (isBound) {
${strictEightsCodes((at) => `lists1[rowBase + ${at}]`, (at) => `lists2[tokenBase + ${at}]`)}
			} else {
${strictEightsCodes((at) => `n_item(places[0], ${at})`, (at) => `n_item(places[2], ${at})`)}
			}
			workgroupBarrier();
${strictEightsInner()}
			i0 += firstWholes;
			i1 += secondWholes;
			i2 += thirdWholes;
			i3 += fourthWholes;
			workgroupBarrier();
		}
		if (local < 64u) {
			let row = rowStart + local;
			rowScales[local] = select(0.0, n_item(places[0], rows * stride + row), row < rows);
		}
		workgroupBarrier();
		sums = mat4x4<f32>(
			vec4<f32>(i0) * rowScales[ownRow * 4u],
			vec4<f32>(i1) * rowScales[ownRow * 4u + 1u],
			vec4<f32>(i2) * rowScales[ownRow * 4u + 2u],
			vec4<f32>(i3) * rowScales[ownRow * 4u + 3u]);
	} else {
		var f0 = vec4<f32>();
		var f1 = vec4<f32>();
		var f2 = vec4<f32>();
		var f3 = vec4<f32>();
		for (var block = 0u; block < blocks; block++) {
			if (isBound) {
${strictEightsCodes((at) => `lists1[rowBase + ${at}]`, (at) => `lists2[tokenBase + ${at}]`)}
			} else {
${strictEightsCodes((at) => `n_item(places[0], ${at})`, (at) => `n_item(places[2], ${at})`)}
			}
			if (local < 64u) {
				let row = rowStart + local;
				rowScales[local] = select(0.0, n_item(places[0], row * stride + rowWords + block), row < rows);
			}
			workgroupBarrier();
${strictEightsInner()}
			f0 += vec4<f32>(firstWholes) * rowScales[ownRow * 4u];
			f1 += vec4<f32>(secondWholes) * rowScales[ownRow * 4u + 1u];
			f2 += vec4<f32>(thirdWholes) * rowScales[ownRow * 4u + 2u];
			f3 += vec4<f32>(fourthWholes) * rowScales[ownRow * 4u + 3u];
			workgroupBarrier();
		}
		sums = mat4x4<f32>(f0, f1, f2, f3);
	}
	let biases = places[5];
	for (var token = 0u; token < 4u; token++) {
		let placedToken = tokenStart + ownToken * 4u + token;
		if (placedToken >= tokens) { break; }
		let scale = n_item(places[2], tokens * words + placedToken);
${packed ? `		let rowBase = rowStart + ownRow * 4u;
		if (rowBase < rows) {
			let at = placedToken * rows + rowBase;
			var v0 = sums[0][token] * scale;
			var v1 = sums[1][token] * scale;
			if (biases > 0u) {
				v0 += n_item(places[4], at % biases);
				v1 += n_item(places[4], (at + 1u) % biases);
			}
			output[at / 2u] = pack_bf16(v0, v1);
			if (rowBase + 2u < rows) {
				var v2 = sums[2][token] * scale;
				var v3 = sums[3][token] * scale;
				if (biases > 0u) {
					v2 += n_item(places[4], (at + 2u) % biases);
					v3 += n_item(places[4], (at + 3u) % biases);
				}
				output[at / 2u + 1u] = pack_bf16(v2, v3);
			}
		}` : `		for (var row = 0u; row < 4u; row++) {
			let placedRow = rowStart + ownRow * 4u + row;
			if (placedRow < rows) {
				let at = placedToken * rows + placedRow;
				var val = sums[row][token] * scale;
				if (biases > 0u) { val += n_item(places[4], at % biases); }
				output[at] = val;
			}
		}`}
	}
}`;
Module['strictShaders']['strict_eights'] = strictEights(false);
Module['strictShaders']['strict_eights_p17'] = strictEights(true);
(Module['strictPacked'] ||= {})['strict_eights'] = 'strict_eights_p17';
(Module['strictCanPack'] ||= {})['strict_eights'] = (places) => (places[9] & 1) === 0;
// the tiles of strict_eights, about 32 billion multiplications in a submit (keeps all SMs saturated)
Module['strictKernelGroups']['strict_eights'] = (places) => {
	const tiles = Math.ceil(places[9] / 64) * Math.ceil(places[13] / 64);
	return [tiles, Math.max(Math.floor(2 ** 35 / (4096 * places[7])), 64)];
};
Module['strictKernelGroups']['strict_eights_p17'] = Module['strictKernelGroups']['strict_eights'];
// Attention.Merged (strict_gpu_attention_web): queries, keys, values, heads, groups, size (up to 128), sequences,
// causal, value offset, value stride; a workgroup of 4 invocations per row of rows tokens (16, or 32 where 32 KB of
// workgroup memory are allowed: keys and values are read half as often) of one head and sequence scores 64 positions at
// a time (each invocation one position for 16 rows), keeps the largest score and the sum of each row (online softmax,
// 4 invocations per row with 16 of the positions each) and adds the weighted values of 2 numbers of 16 rows per
// invocation, no scores are stored; keys go through workgroup memory 16 dimensions at a time (coalesced, XOR swizzled)
const strictAttention = (rows, half = false, subgroups = false, packed = false) => {
	const invocations = rows * 4, quads = rows / 4;
	const sc = half ? 'f16' : 'f32', vBatch = half ? Math.max(rows / 4, 4) : rows / 2;
	const kLoadCase = (buf) => `				for (var part = 0u; part < ${32 * rows / invocations}u; part++) {
					let element = part * ${invocations}u + local;
					let row = element / ${rows / 2}u;
					let dimension = element % ${rows / 2}u * 2u;
					let keyPosition = start + row;
					var pair = vec2<f32>();
					if (keyPosition < visible && chunk + dimension < size) {
						let raw = bitcast<u32>(${buf}[kBase + ((keyPosition * sequences * keyGroups + keyGroup) * size + chunk + dimension) / 2u]);
						pair = select(vec2<f32>(bitcast<f32>(raw << 16u), bitcast<f32>(raw & 0xffff0000u)), unpack2x16float(raw), kF16);
					}
					let slot = dimension * 64u + (row ^ dimension);
					scored[slot / 4u][slot % 4u] = ${sc}(pair.x);
					let next = (dimension + 1u) * 64u + (row ^ (dimension + 1u));
					scored[next / 4u][next % 4u] = ${sc}(pair.y);
				}`;
	const vLoadCase = (buf) => `				for (var p = local; p < totalWords; p += ${invocations}u) {
					let jp = p / 64u;
					let cp = p % 64u;
					var raw = 0u;
					if (cp * 2u < size) {
						raw = bitcast<u32>(${buf}[vBase + (valueOffset + keyGroup * size + (start + sub + jp) * valueStride) / 2u + cp]);
					}
					valued[p] = raw;
				}`;
	const listSwitch = (bindVar, loopGen) => `			switch (${bindVar}) {
				case 1u: {
${loopGen('lists1')}
				}
				case 2u: {
${loopGen('lists2')}
				}
				case 3u: {
${loopGen('lists3')}
				}
				case 4u: {
${loopGen('lists4')}
				}
				default: {
${loopGen('numbers')}
				}
			}`;
	return (half ? 'enable f16;\n' : '') + (subgroups ? 'enable subgroups;\n' : '') + strictKernelBindings + `
var<workgroup> asked: array<vec4<${sc}>, ${128 * quads}>;
var<workgroup> scored: array<vec4<${sc}>, ${64 * quads}>;
var<workgroup> valued: array<u32, ${vBatch * 64}>;
var<workgroup> largests: array<f32, ${rows}>;
var<workgroup> rescales: array<vec4<f32>, ${quads}>;
var<workgroup> totals: array<f32, ${rows}>;
${subgroups ? '' : `var<workgroup> maxima: array<f32, ${invocations}>;\nvar<workgroup> partials: array<f32, ${invocations}>;\n`}@compute @workgroup_size(${invocations})
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
	let tokenBlocks = (tokens + ${rows - 1}u) / ${rows}u;
	let block = work.x + group.x + group.y * groups.x;
	if (block >= sequences * tokenBlocks * heads) { return; }
	let firstToken = block % tokenBlocks * ${rows}u;
	let head = block / tokenBlocks % heads;
	let sequence = block / (tokenBlocks * heads);
	let keyGroup = sequence * keyGroups + head * keyGroups / heads;
	let fastK = (places[2] & 0x80000000u) != 0u && ((sequences * keyGroups * size) & 1u) == 0u;
	let kF16 = (places[2] & 0x08000000u) != 0u;
	let kBind = (places[2] >> 28u) & 7u;
	let kBase = places[2] & 0x0fffffffu;
	let fastV = (places[4] & 0x80000000u) != 0u && ((valueOffset | size | valueStride) & 1u) == 0u && size <= 128u;
	let isF16V = (places[4] & 0x08000000u) != 0u;
	let vBind = (places[4] >> 28u) & 7u;
	let vBase = places[4] & 0x0fffffffu;
	for (var at = local; at < ${rows}u * size; at += ${invocations}u) {
		let row = at / size;
		let token = firstToken + row;
		var query = 0.0;
		if (token < tokens) { query = n_item(places[0], ((sequence * tokens + token) * heads + head) * size + at % size); }
		asked[at % size * ${quads}u + row / 4u][row % 4u] = ${sc}(query / sqrt(f32(size)));
	}
	if (local < ${rows}u) {
		largests[local] = -3.0e38;
		totals[local] = 0.0;
	}
	let lastToken = min(firstToken + ${rows - 1}u, tokens - 1u);
	let visible = select(positions, positions - tokens + lastToken + 1u, causal);
	let half = local / 64u;
	let own = local % 64u;
	let first = half * 4u;
	let lasts = vec4<u32>(0u, 1u, 2u, 3u) + firstToken + half * 16u +
		select(positions - 1u - firstToken, positions - tokens, causal);
	var firstLow = vec4<f32>();
	var secondLow = vec4<f32>();
	var thirdLow = vec4<f32>();
	var fourthLow = vec4<f32>();
	var firstHigh = vec4<f32>();
	var secondHigh = vec4<f32>();
	var thirdHigh = vec4<f32>();
	var fourthHigh = vec4<f32>();
	for (var start = 0u; start < visible; start += 64u) {
		workgroupBarrier();
		let position = start + own;
		var firstScores = vec4<f32>();
		var secondScores = vec4<f32>();
		var thirdScores = vec4<f32>();
		var fourthScores = vec4<f32>();
		for (var chunk = 0u; chunk < size; chunk += ${rows}u) {
			workgroupBarrier();
			if (fastK) {
${listSwitch('kBind', kLoadCase)}
			} else {
				for (var part = 0u; part < ${32 * rows / invocations}u; part++) {
					let element = part * ${invocations}u + local;
					let row = element / ${rows / 2}u;
					let dimension = element % ${rows / 2}u * 2u;
					let keyPosition = start + row;
					var pair = vec2<f32>();
					if (keyPosition < visible && chunk + dimension < size) {
						pair = n_pair(places[2], (keyPosition * sequences * keyGroups + keyGroup) * size + chunk + dimension);
					}
					let slot = dimension * 64u + (row ^ dimension);
					scored[slot / 4u][slot % 4u] = ${sc}(pair.x);
					let next = (dimension + 1u) * 64u + (row ^ (dimension + 1u));
					scored[next / 4u][next % 4u] = ${sc}(pair.y);
				}
			}
			workgroupBarrier();
			let dims = min(${rows}u, size - chunk);
			for (var dimension = 0u; dimension < dims; dimension++) {
				let slot = dimension * 64u + (own ^ dimension);
				let key = vec4<f32>(f32(scored[slot / 4u][slot % 4u]));
				let at = (chunk + dimension) * ${quads}u + first;
				firstScores = fma(vec4<f32>(asked[at]), key, firstScores);
				secondScores = fma(vec4<f32>(asked[at + 1u]), key, secondScores);
				thirdScores = fma(vec4<f32>(asked[at + 2u]), key, thirdScores);
				fourthScores = fma(vec4<f32>(asked[at + 3u]), key, fourthScores);
			}
		}
		workgroupBarrier();
		let hidden = vec4<${sc}>(${half ? '-65000.0h' : '-3.0e38'});
		let placedScores = own * ${quads}u + first;
		let seen = vec4<bool>(position < visible);
		scored[placedScores] = select(hidden, vec4<${sc}>(firstScores), (vec4<u32>(position) <= lasts) & seen);
		scored[placedScores + 1u] = select(hidden, vec4<${sc}>(secondScores), (vec4<u32>(position) <= lasts + 4u) & seen);
		scored[placedScores + 2u] = select(hidden, vec4<${sc}>(thirdScores), (vec4<u32>(position) <= lasts + 8u) & seen);
		scored[placedScores + 3u] = select(hidden, vec4<${sc}>(fourthScores), (vec4<u32>(position) <= lasts + 12u) & seen);
		workgroupBarrier();
		let row = local / 4u;
		let quad = row / 4u;
		let lane = row % 4u;
		let begin = local % 4u * 16u;
		var largest = largests[row];
		for (var at = begin; at < begin + 16u; at++) { largest = max(largest, f32(scored[at * ${quads}u + quad][lane])); }
${subgroups ? `		largest = max(largest, subgroupShuffleXor(largest, 1u));
		largest = max(largest, subgroupShuffleXor(largest, 2u));` : `		maxima[local] = largest;
		workgroupBarrier();
		largest = max(max(maxima[row * 4u], maxima[row * 4u + 1u]), max(maxima[row * 4u + 2u], maxima[row * 4u + 3u]));`}
		var total = 0.0;
		for (var at = begin; at < begin + 16u; at++) {
			let score = f32(scored[at * ${quads}u + quad][lane]);
			let share = select(exp(score - largest), 0.0, score < ${half ? '-60000.0' : '-1.0e38'});
			scored[at * ${quads}u + quad][lane] = ${sc}(share);
			total += share;
		}
${subgroups ? `		total += subgroupShuffleXor(total, 1u);
		total += subgroupShuffleXor(total, 2u);
		if (begin == 0u) {
			let rescale = exp(largests[row] - largest);
			rescales[quad][lane] = rescale;
			totals[row] = totals[row] * rescale + total;
			largests[row] = largest;
		}` : `		partials[local] = total;
		workgroupBarrier();
		if (begin == 0u) {
			let rescale = exp(largests[row] - largest);
			rescales[quad][lane] = rescale;
			totals[row] = totals[row] * rescale + partials[local] + partials[local + 1u] + partials[local + 2u] +
				partials[local + 3u];
			largests[row] = largest;
		}`}
		workgroupBarrier();
		firstLow *= rescales[first];
		secondLow *= rescales[first + 1u];
		thirdLow *= rescales[first + 2u];
		fourthLow *= rescales[first + 3u];
		firstHigh *= rescales[first];
		secondHigh *= rescales[first + 1u];
		thirdHigh *= rescales[first + 2u];
		fourthHigh *= rescales[first + 3u];
		let count = min(64u, visible - start);
		if (fastV) {
			for (var sub = 0u; sub < count; sub += ${vBatch}u) {
				let batch = min(${vBatch}u, count - sub);
				let totalWords = batch * 64u;
${listSwitch('vBind', vLoadCase)}
				workgroupBarrier();
				for (var at = 0u; at < batch; at++) {
					let raw = valued[at * 64u + own];
					let pair = select(vec2<f32>(bitcast<f32>(raw << 16u), bitcast<f32>(raw & 0xffff0000u)),
						unpack2x16float(raw), isF16V);
					let low = vec4<f32>(pair.x);
					let high = vec4<f32>(select(0.0, pair.y, own * 2u + 1u < size));
					let weights = (sub + at) * ${quads}u + first;
					firstLow = fma(vec4<f32>(scored[weights]), low, firstLow);
					secondLow = fma(vec4<f32>(scored[weights + 1u]), low, secondLow);
					thirdLow = fma(vec4<f32>(scored[weights + 2u]), low, thirdLow);
					fourthLow = fma(vec4<f32>(scored[weights + 3u]), low, fourthLow);
					firstHigh = fma(vec4<f32>(scored[weights]), high, firstHigh);
					secondHigh = fma(vec4<f32>(scored[weights + 1u]), high, secondHigh);
					thirdHigh = fma(vec4<f32>(scored[weights + 2u]), high, thirdHigh);
					fourthHigh = fma(vec4<f32>(scored[weights + 3u]), high, fourthHigh);
				}
				workgroupBarrier();
			}
		} else {
			for (var at = 0u; at < count; at++) {
				let valueStart = valueOffset + keyGroup * size + (start + at) * valueStride;
				var pair = vec2<f32>();
				if (own * 2u < size) { pair = n_pair(places[4], valueStart + own * 2u); }
				let low = vec4<f32>(pair.x);
				let high = vec4<f32>(select(0.0, pair.y, own * 2u + 1u < size));
				let weights = at * ${quads}u + first;
				firstLow = fma(vec4<f32>(scored[weights]), low, firstLow);
				secondLow = fma(vec4<f32>(scored[weights + 1u]), low, secondLow);
				thirdLow = fma(vec4<f32>(scored[weights + 2u]), low, thirdLow);
				fourthLow = fma(vec4<f32>(scored[weights + 3u]), low, fourthLow);
				firstHigh = fma(vec4<f32>(scored[weights]), high, firstHigh);
				secondHigh = fma(vec4<f32>(scored[weights + 1u]), high, secondHigh);
				thirdHigh = fma(vec4<f32>(scored[weights + 2u]), high, thirdHigh);
				fourthHigh = fma(vec4<f32>(scored[weights + 3u]), high, fourthHigh);
			}
		}
	}
	workgroupBarrier();
	let lows = mat4x4<f32>(firstLow, secondLow, thirdLow, fourthLow);
	let highs = mat4x4<f32>(firstHigh, secondHigh, thirdHigh, fourthHigh);
	for (var row = 0u; row < 16u; row++) {
		let token = firstToken + half * 16u + row;
		if (token >= tokens) { break; }
		let placed = ((sequence * tokens + token) * heads + head) * size;
		let total = totals[half * 16u + row];
${packed ? `		if (own * 2u + 1u < size) {
			output[(placed + own * 2u) / 2u] = pack_bf16(lows[row / 4u][row % 4u] / total, highs[row / 4u][row % 4u] / total);
		}` : `		if (own * 2u < size) { output[placed + own * 2u] = lows[row / 4u][row % 4u] / total; }
		if (own * 2u + 1u < size) { output[placed + own * 2u + 1u] = highs[row / 4u][row % 4u] / total; }`}
	}
}`;
};
const strictAttention64 = (subgroups = false, packed = false) => {
	const kLoadCase = (buf) => `				for (var part = 0u; part < 16u; part++) {
					let element = part * 128u + local;
					let row = element / 32u;
					let dimension = element % 32u * 2u;
					let keyPosition = start + row;
					var pair = vec2<f32>();
					if (keyPosition < visible && chunk + dimension < size) {
						let raw = bitcast<u32>(${buf}[kBase + ((keyPosition * sequences * keyGroups + keyGroup) * size + chunk + dimension) / 2u]);
						pair = select(vec2<f32>(bitcast<f32>(raw << 16u), bitcast<f32>(raw & 0xffff0000u)), unpack2x16float(raw), kF16);
					}
					let slot = dimension * 64u + (row ^ dimension);
					scored[slot / 4u][slot % 4u] = f16(pair.x);
					let next = (dimension + 1u) * 64u + (row ^ (dimension + 1u));
					scored[next / 4u][next % 4u] = f16(pair.y);
				}`;
	const vLoadCase = (buf) => `				for (var p = local; p < totalWords; p += 128u) {
					let jp = p / 64u;
					let cp = p % 64u;
					var raw = 0u;
					if (cp * 2u < size) {
						raw = bitcast<u32>(${buf}[vBase + (valueOffset + keyGroup * size + (start + sub + jp) * valueStride) / 2u + cp]);
					}
					valued[p] = raw;
				}`;
	const listSwitch = (bindVar, loopGen) => `			switch (${bindVar}) {
				case 1u: {
${loopGen('lists1')}
				}
				case 2u: {
${loopGen('lists2')}
				}
				case 3u: {
${loopGen('lists3')}
				}
				case 4u: {
${loopGen('lists4')}
				}
				default: {
${loopGen('numbers')}
				}
			}`;
	return 'enable f16;\n' + (subgroups ? 'enable subgroups;\n' : '') + strictKernelBindings + `
var<workgroup> asked: array<vec4<f16>, 2048>;
var<workgroup> scored: array<vec4<f16>, 1024>;
var<workgroup> valued: array<u32, 1024>;
var<workgroup> largests: array<f32, 64>;
var<workgroup> rescales: array<vec4<f32>, 16>;
var<workgroup> totals: array<f32, 64>;
${subgroups ? '' : 'var<workgroup> maxima: array<f32, 128>;\nvar<workgroup> partials: array<f32, 128>;\n'}@compute @workgroup_size(128)
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
	let tokenBlocks = (tokens + 63u) / 64u;
	let block = work.x + group.x + group.y * groups.x;
	if (block >= sequences * tokenBlocks * heads) { return; }
	let firstToken = block % tokenBlocks * 64u;
	let head = block / tokenBlocks % heads;
	let sequence = block / (tokenBlocks * heads);
	let keyGroup = sequence * keyGroups + head * keyGroups / heads;
	let fastK = (places[2] & 0x80000000u) != 0u && ((sequences * keyGroups * size) & 1u) == 0u;
	let kF16 = (places[2] & 0x08000000u) != 0u;
	let kBind = (places[2] >> 28u) & 7u;
	let kBase = places[2] & 0x0fffffffu;
	let fastV = (places[4] & 0x80000000u) != 0u && ((valueOffset | size | valueStride) & 1u) == 0u && size <= 128u;
	let isF16V = (places[4] & 0x08000000u) != 0u;
	let vBind = (places[4] >> 28u) & 7u;
	let vBase = places[4] & 0x0fffffffu;
	let invSqrt = 1.0 / sqrt(f32(size));
	for (var at = local; at < 64u * size; at += 128u) {
		let row = at / size;
		let token = firstToken + row;
		var query = 0.0;
		if (token < tokens) { query = n_item(places[0], ((sequence * tokens + token) * heads + head) * size + at % size); }
		asked[at % size * 16u + row / 4u][row % 4u] = f16(query * invSqrt);
	}
	if (local < 64u) {
		largests[local] = -3.0e38;
		totals[local] = 0.0;
	}
	let lastToken = min(firstToken + 63u, tokens - 1u);
	let visible = select(positions, positions - tokens + lastToken + 1u, causal);
	let quarter = local / 32u;
	let own = local % 32u;
	let first = quarter * 4u;
	let lasts = vec4<u32>(0u, 1u, 2u, 3u) + firstToken + quarter * 16u +
		select(positions - 1u - firstToken, positions - tokens, causal);
	var d0l = vec4<f32>(); var d0h = vec4<f32>(); var d1l = vec4<f32>(); var d1h = vec4<f32>();
	var d2l = vec4<f32>(); var d2h = vec4<f32>(); var d3l = vec4<f32>(); var d3h = vec4<f32>();
	var d0b = vec4<f32>(); var d0c = vec4<f32>(); var d1b = vec4<f32>(); var d1c = vec4<f32>();
	var d2b = vec4<f32>(); var d2c = vec4<f32>(); var d3b = vec4<f32>(); var d3c = vec4<f32>();
	for (var start = 0u; start < visible; start += 64u) {
		workgroupBarrier();
		let posA = start + own;
		let posB = posA + 32u;
		var sA0 = vec4<f16>(); var sA1 = vec4<f16>(); var sA2 = vec4<f16>(); var sA3 = vec4<f16>();
		var sB0 = vec4<f16>(); var sB1 = vec4<f16>(); var sB2 = vec4<f16>(); var sB3 = vec4<f16>();
		for (var chunk = 0u; chunk < size; chunk += 64u) {
			workgroupBarrier();
			if (fastK) {
${listSwitch('kBind', kLoadCase)}
			} else {
				for (var part = 0u; part < 16u; part++) {
					let element = part * 128u + local;
					let row = element / 32u;
					let dimension = element % 32u * 2u;
					let keyPosition = start + row;
					var pair = vec2<f32>();
					if (keyPosition < visible && chunk + dimension < size) {
						pair = n_pair(places[2], (keyPosition * sequences * keyGroups + keyGroup) * size + chunk + dimension);
					}
					let slot = dimension * 64u + (row ^ dimension);
					scored[slot / 4u][slot % 4u] = f16(pair.x);
					let next = (dimension + 1u) * 64u + (row ^ (dimension + 1u));
					scored[next / 4u][next % 4u] = f16(pair.y);
				}
			}
			workgroupBarrier();
			let dims = min(64u, size - chunk);
			for (var dimension = 0u; dimension < dims; dimension++) {
				let slotA = dimension * 64u + (own ^ dimension);
				let kA = vec4<f16>(scored[slotA / 4u][slotA % 4u]);
				let slotB = dimension * 64u + ((own + 32u) ^ dimension);
				let kB = vec4<f16>(scored[slotB / 4u][slotB % 4u]);
				let at = (chunk + dimension) * 16u + first;
				let q0 = asked[at];
				let q1 = asked[at + 1u];
				let q2 = asked[at + 2u];
				let q3 = asked[at + 3u];
				sA0 = fma(q0, kA, sA0); sA1 = fma(q1, kA, sA1); sA2 = fma(q2, kA, sA2); sA3 = fma(q3, kA, sA3);
				sB0 = fma(q0, kB, sB0); sB1 = fma(q1, kB, sB1); sB2 = fma(q2, kB, sB2); sB3 = fma(q3, kB, sB3);
			}
		}
		workgroupBarrier();
		let hidden = vec4<f16>(-65000.0h);
		let seenA = vec4<bool>(posA < visible);
		let placedA = own * 16u + first;
		scored[placedA] = select(hidden, sA0, (vec4<u32>(posA) <= lasts) & seenA);
		scored[placedA + 1u] = select(hidden, sA1, (vec4<u32>(posA) <= lasts + 4u) & seenA);
		scored[placedA + 2u] = select(hidden, sA2, (vec4<u32>(posA) <= lasts + 8u) & seenA);
		scored[placedA + 3u] = select(hidden, sA3, (vec4<u32>(posA) <= lasts + 12u) & seenA);
		let seenB = vec4<bool>(posB < visible);
		let placedB = (own + 32u) * 16u + first;
		scored[placedB] = select(hidden, sB0, (vec4<u32>(posB) <= lasts) & seenB);
		scored[placedB + 1u] = select(hidden, sB1, (vec4<u32>(posB) <= lasts + 4u) & seenB);
		scored[placedB + 2u] = select(hidden, sB2, (vec4<u32>(posB) <= lasts + 8u) & seenB);
		scored[placedB + 3u] = select(hidden, sB3, (vec4<u32>(posB) <= lasts + 12u) & seenB);
		workgroupBarrier();
		let row = local / 2u;
		let quad = row / 4u;
		let lane = row % 4u;
		let begin = local % 2u * 32u;
		var largest = largests[row];
		for (var at = begin; at < begin + 32u; at++) { largest = max(largest, f32(scored[at * 16u + quad][lane])); }
${subgroups ? '		largest = max(largest, subgroupShuffleXor(largest, 1u));' : `		maxima[local] = largest;
		workgroupBarrier();
		largest = max(maxima[row * 2u], maxima[row * 2u + 1u]);`}
		var total = 0.0;
		for (var at = begin; at < begin + 32u; at++) {
			let score = f32(scored[at * 16u + quad][lane]);
			let share = select(exp(score - largest), 0.0, score < -60000.0);
			scored[at * 16u + quad][lane] = f16(share);
			total += share;
		}
${subgroups ? `		total += subgroupShuffleXor(total, 1u);
		if (begin == 0u) {
			let rescale = exp(largests[row] - largest);
			rescales[quad][lane] = rescale;
			totals[row] = totals[row] * rescale + total;
			largests[row] = largest;
		}` : `		partials[local] = total;
		workgroupBarrier();
		if (begin == 0u) {
			let rescale = exp(largests[row] - largest);
			rescales[quad][lane] = rescale;
			totals[row] = totals[row] * rescale + partials[local] + partials[local + 1u];
			largests[row] = largest;
		}`}
		workgroupBarrier();
		let r0 = rescales[first]; let r1 = rescales[first + 1u];
		let r2 = rescales[first + 2u]; let r3 = rescales[first + 3u];
		var t0l = vec4<f16>(); var t0h = vec4<f16>(); var t1l = vec4<f16>(); var t1h = vec4<f16>();
		var t2l = vec4<f16>(); var t2h = vec4<f16>(); var t3l = vec4<f16>(); var t3h = vec4<f16>();
		var t0b = vec4<f16>(); var t0c = vec4<f16>(); var t1b = vec4<f16>(); var t1c = vec4<f16>();
		var t2b = vec4<f16>(); var t2c = vec4<f16>(); var t3b = vec4<f16>(); var t3c = vec4<f16>();
		let count = min(64u, visible - start);
		if (fastV) {
			for (var sub = 0u; sub < count; sub += 16u) {
				let batch = min(16u, count - sub);
				let totalWords = batch * 64u;
${listSwitch('vBind', vLoadCase)}
				workgroupBarrier();
				for (var at = 0u; at < batch; at++) {
					let raw0 = valued[at * 64u + own * 2u];
					let raw1 = valued[at * 64u + own * 2u + 1u];
					let p0 = select(vec2<f32>(bitcast<f32>(raw0 << 16u), bitcast<f32>(raw0 & 0xffff0000u)),
						unpack2x16float(raw0), isF16V);
					let p1 = select(vec2<f32>(bitcast<f32>(raw1 << 16u), bitcast<f32>(raw1 & 0xffff0000u)),
						unpack2x16float(raw1), isF16V);
					let v0 = vec4<f16>(f16(p0.x));
					let v1 = vec4<f16>(f16(select(0.0, p0.y, own * 4u + 1u < size)));
					let v2 = vec4<f16>(f16(select(0.0, p1.x, own * 4u + 2u < size)));
					let v3 = vec4<f16>(f16(select(0.0, p1.y, own * 4u + 3u < size)));
					let weights = (sub + at) * 16u + first;
					let w0 = scored[weights];
					let w1 = scored[weights + 1u];
					let w2 = scored[weights + 2u];
					let w3 = scored[weights + 3u];
					t0l = fma(w0, v0, t0l); t1l = fma(w1, v0, t1l); t2l = fma(w2, v0, t2l); t3l = fma(w3, v0, t3l);
					t0h = fma(w0, v1, t0h); t1h = fma(w1, v1, t1h); t2h = fma(w2, v1, t2h); t3h = fma(w3, v1, t3h);
					t0b = fma(w0, v2, t0b); t1b = fma(w1, v2, t1b); t2b = fma(w2, v2, t2b); t3b = fma(w3, v2, t3b);
					t0c = fma(w0, v3, t0c); t1c = fma(w1, v3, t1c); t2c = fma(w2, v3, t2c); t3c = fma(w3, v3, t3c);
				}
				workgroupBarrier();
			}
		} else {
			for (var at = 0u; at < count; at++) {
				let valueStart = valueOffset + keyGroup * size + (start + at) * valueStride;
				var p0 = vec2<f32>();
				var p1 = vec2<f32>();
				if (own * 4u < size) { p0 = n_pair(places[4], valueStart + own * 4u); }
				if (own * 4u + 2u < size) { p1 = n_pair(places[4], valueStart + own * 4u + 2u); }
				let v0 = vec4<f16>(f16(p0.x));
				let v1 = vec4<f16>(f16(select(0.0, p0.y, own * 4u + 1u < size)));
				let v2 = vec4<f16>(f16(select(0.0, p1.x, own * 4u + 2u < size)));
				let v3 = vec4<f16>(f16(select(0.0, p1.y, own * 4u + 3u < size)));
				let weights = at * 16u + first;
				let w0 = scored[weights];
				let w1 = scored[weights + 1u];
				let w2 = scored[weights + 2u];
				let w3 = scored[weights + 3u];
				t0l = fma(w0, v0, t0l); t1l = fma(w1, v0, t1l); t2l = fma(w2, v0, t2l); t3l = fma(w3, v0, t3l);
				t0h = fma(w0, v1, t0h); t1h = fma(w1, v1, t1h); t2h = fma(w2, v1, t2h); t3h = fma(w3, v1, t3h);
				t0b = fma(w0, v2, t0b); t1b = fma(w1, v2, t1b); t2b = fma(w2, v2, t2b); t3b = fma(w3, v2, t3b);
				t0c = fma(w0, v3, t0c); t1c = fma(w1, v3, t1c); t2c = fma(w2, v3, t2c); t3c = fma(w3, v3, t3c);
			}
		}
		d0l = fma(d0l, r0, vec4<f32>(t0l)); d1l = fma(d1l, r1, vec4<f32>(t1l)); d2l = fma(d2l, r2, vec4<f32>(t2l)); d3l = fma(d3l, r3, vec4<f32>(t3l));
		d0h = fma(d0h, r0, vec4<f32>(t0h)); d1h = fma(d1h, r1, vec4<f32>(t1h)); d2h = fma(d2h, r2, vec4<f32>(t2h)); d3h = fma(d3h, r3, vec4<f32>(t3h));
		d0b = fma(d0b, r0, vec4<f32>(t0b)); d1b = fma(d1b, r1, vec4<f32>(t1b)); d2b = fma(d2b, r2, vec4<f32>(t2b)); d3b = fma(d3b, r3, vec4<f32>(t3b));
		d0c = fma(d0c, r0, vec4<f32>(t0c)); d1c = fma(d1c, r1, vec4<f32>(t1c)); d2c = fma(d2c, r2, vec4<f32>(t2c)); d3c = fma(d3c, r3, vec4<f32>(t3c));
	}
	workgroupBarrier();
	let a0 = mat4x4<f32>(d0l, d1l, d2l, d3l);
	let a1 = mat4x4<f32>(d0h, d1h, d2h, d3h);
	let a2 = mat4x4<f32>(d0b, d1b, d2b, d3b);
	let a3 = mat4x4<f32>(d0c, d1c, d2c, d3c);
	for (var row = 0u; row < 16u; row++) {
		let token = firstToken + quarter * 16u + row;
		if (token >= tokens) { break; }
		let placed = ((sequence * tokens + token) * heads + head) * size;
		let total = totals[quarter * 16u + row];
		let col = own * 4u;
${packed ? `		if (col + 1u < size) { output[(placed + col) / 2u] = pack_bf16(a0[row / 4u][row % 4u] / total, a1[row / 4u][row % 4u] / total); }
		if (col + 3u < size) { output[(placed + col) / 2u + 1u] = pack_bf16(a2[row / 4u][row % 4u] / total, a3[row / 4u][row % 4u] / total); }` : `		if (col < size) { output[placed + col] = a0[row / 4u][row % 4u] / total; }
		if (col + 1u < size) { output[placed + col + 1u] = a1[row / 4u][row % 4u] / total; }
		if (col + 2u < size) { output[placed + col + 2u] = a2[row / 4u][row % 4u] / total; }
		if (col + 3u < size) { output[placed + col + 3u] = a3[row / 4u][row % 4u] / total; }`}
	}
}`;
};
const strictAttentionGroups = (rows) => (places) => {
	const tokens = places[1] / places[7] / places[11] / places[13];
	const blocks = places[13] * Math.ceil(tokens / rows) * places[7];
	const positions = places[3] / places[9] / places[11] / places[13];
	return [blocks, Math.max(Math.floor(2 ** 36 / (2 * rows * positions * places[11])), 1)];
};
for (const [key, rows, fn] of [
	['strict_attention', 16, (sg, pk) => strictAttention(16, false, sg, pk)],
	['strict_attention32', 32, (sg, pk) => strictAttention(32, false, sg, pk)],
	['strict_attention64_f16', 64, (sg, pk) => strictAttention64(sg, pk)],
]) {
	const grp = strictAttentionGroups(rows);
	const sg = key + '_sg', p17 = key + '_p17', sgp17 = sg + '_p17';
	Module['strictShaders'][key] = fn(false, false);
	Module['strictShaders'][sg] = fn(true, false);
	Module['strictShaders'][p17] = fn(false, true);
	Module['strictShaders'][sgp17] = fn(true, true);
	(Module['strictSubgroups'] ||= {})[key] = sg;
	(Module['strictPacked'] ||= {})[key] = p17;
	Module['strictPacked'][sg] = sgp17;
	(Module['strictCanPack'] ||= {})[key] = Module['strictCanPack'][sg] = (places) => (places[11] & 1) === 0;
	Module['strictKernelGroups'][key] = Module['strictKernelGroups'][sg] = Module['strictKernelGroups'][p17] = Module['strictKernelGroups'][sgp17] = grp;
}
(Module['strictLarger'] ||= {})['strict_attention'] = 'strict_attention32';
(Module['strictHalf'] ||= {})['strict_attention32'] = 'strict_attention64_f16';
// Batch.Sums of 32 bit rows (strict_gpu_tokens_web): sums of strict_products, biases: every sum plus its bias
Module['strictShaders']['strict_biased'] = strictKernelBindings + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let at = id.x + id.y * groups.x * 64u;
	if (at >= places[4]) { return; }
	output[at] = n_item(places[0], at) + n_item(places[2], at % places[3]);
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
// the values of a convolution normed with affine and activated once (strict_gpu_convolution), not in each of its
// tiles: values, affine, channels, Activation kind (5 swish, 2 relu, 6 linear)
const strictNormed = (packed = false) => strictKernelBindings + `
fn strict_norm_at(at: u32) -> f32 {
	let channel = at % places[5];
	let normed = n_item(places[0], at) * n_item(places[2], channel) + n_item(places[2], places[5] + channel);
	return select(select(normed, max(normed, 0.0), places[7] == 2u), normed / (1.0 + exp(-normed)), places[7] == 5u);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let at = id.x + id.y * groups.x * 64u;
${packed ? `	let count = places[8];
	if (at * 2u >= count) { return; }
	let second = select(0.0, strict_norm_at(at * 2u + 1u), at * 2u + 1u < count);
	output[at] = pack_bf16(strict_norm_at(at * 2u), second);` : `	if (at >= places[8]) { return; }
	output[at] = strict_norm_at(at);`}
}`;
Module['strictShaders']['strict_normed'] = strictNormed(false);
Module['strictShaders']['strict_normed_p17'] = strictNormed(true);
Module['strictPacked']['strict_normed'] = 'strict_normed_p17';
// Convrot.Rotated (strict_gpu_hadamard): values; a workgroup of 64 invocations per group of 256 values runs the 4
// radix-4 butterflies (steps 1, 4, 16, 64, a butterfly of 4 values each) in workgroup memory, 32 bit floats in between
const strictHadamard = (packed = false) => strictKernelBindings + `
var<workgroup> rotated: array<f32, 256>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>) {
	let offset = (work.x + group.x + group.y * groups.x) * 256u;
	if (offset >= places[2]) { return; }
	for (var at = local; at < 256u; at += 64u) { rotated[at] = n_item(places[0], offset + at); }
	workgroupBarrier();
	for (var step = 1u; step < 256u; step *= 4u) {
		let start = local / step * 4u * step + local % step;
		let first = rotated[start];
		let second = rotated[start + step];
		let third = rotated[start + 2u * step];
		let fourth = rotated[start + 3u * step];
		rotated[start] = 0.5 * (first + second + third - fourth);
		rotated[start + step] = 0.5 * (first + second - third + fourth);
		rotated[start + 2u * step] = 0.5 * (first - second + third + fourth);
		rotated[start + 3u * step] = 0.5 * (second + third + fourth - first);
		workgroupBarrier();
	}
${packed ? `	for (var pair = local; pair < 128u; pair += 64u) {
		output[offset / 2u + pair] = pack_bf16(rotated[pair * 2u], rotated[pair * 2u + 1u]);
	}` : `	for (var at = local; at < 256u; at += 64u) { output[offset + at] = rotated[at]; }`}
}`;
Module['strictShaders']['strict_hadamard'] = strictHadamard(false);
Module['strictShaders']['strict_hadamard_p17'] = strictHadamard(true);
Module['strictPacked']['strict_hadamard'] = 'strict_hadamard_p17';
Module['strictKernelGroups']['strict_hadamard'] = Module['strictKernelGroups']['strict_hadamard_p17'] = (places) => [places[2] / 256, places[2] / 256];
const strictHeadRotated = (subgroups = false, packed = false) => (subgroups ? 'enable subgroups;\n' : '') + strictKernelBindings + `
var<workgroup> headWx: array<f32, 128>;
var<workgroup> headSq: array<f32, ${subgroups ? 32 : 128}>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>${subgroups ? ', @builtin(subgroup_invocation_id) sgLane: u32, @builtin(subgroup_size) sgSize: u32' : ''}) {
	let headIdx = work.x + group.x + group.y * groups.x;
	let outOffset = headIdx * 128u;
	if (outOffset >= places[14]) { return; }
	let width = places[5];
	let heads = width / 128u;
	let token = headIdx / heads;
	let head = headIdx % heads;
	let inOffset = token * 3u * width + places[3] + head * 128u;
	let x = n_item(places[0], inOffset + local);
	let wx = x * n_item(places[6], local);
	headWx[local] = wx;
${subgroups ? `	let sgSum = subgroupAdd(x * x);
	if (sgLane == 0u) { headSq[local / sgSize] = sgSum; }
	workgroupBarrier();
	var sqTotal = 0.0;
	let sgCount = (128u + sgSize - 1u) / sgSize;
	for (var i = 0u; i < sgCount; i++) { sqTotal += headSq[i]; }
	let scale = inverseSqrt(sqTotal / 128.0 + 1e-6);` : `	headSq[local] = x * x;
	workgroupBarrier();
	for (var half = 64u; half > 0u; half /= 2u) {
		if (local < half) { headSq[local] += headSq[local + half]; }
		workgroupBarrier();
	}
	let scale = inverseSqrt(headSq[0] / 128.0 + 1e-6);`}
	let partner = headWx[(local + 64u) & 127u];
	let rotBase = token * 256u;
	let c = n_item(places[8], rotBase + local);
	let s = n_item(places[8], rotBase + 128u + local);
	let outVal = (wx * c + partner * s) * scale;
${packed ? (subgroups ? `	let oddVal = subgroupShuffleXor(outVal, 1u);
	if ((local & 1u) == 0u) { output[headIdx * 64u + local / 2u] = pack_bf16(outVal, oddVal); }` : `	workgroupBarrier();
	headWx[local] = outVal;
	workgroupBarrier();
	if (local < 64u) { output[headIdx * 64u + local] = pack_bf16(headWx[local * 2u], headWx[local * 2u + 1u]); }`) : `	output[outOffset + local] = outVal;`}
}`;
for (const [key, sg] of [['strict_device_head_rotated', false], ['strict_device_head_rotated_sg', true]]) {
	const p17 = key + '_p17';
	Module['strictShaders'][key] = strictHeadRotated(sg, false);
	Module['strictShaders'][p17] = strictHeadRotated(sg, true);
	Module['strictPacked'][key] = p17;
	Module['strictKernelGroups'][key] = Module['strictKernelGroups'][p17] = (places) => [places[14] / 128, places[14] / 128];
}
Module['strictSubgroups']['strict_device_head_rotated'] = 'strict_device_head_rotated_sg';
const strictTokenModulated = (subgroups = false, packed = false) => (subgroups ? 'enable subgroups;\n' : '') + strictKernelBindings + `
var<workgroup> modPartials: array<f32, ${subgroups ? 64 : 256}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>,
	@builtin(num_workgroups) groups: vec3<u32>${subgroups ? ', @builtin(subgroup_invocation_id) sgLane: u32, @builtin(subgroup_size) sgSize: u32' : ''}) {
	let token = work.x + group.x + group.y * groups.x;
	let width = places[3];
	let tokenStart = token * width;
	if (tokenStart >= places[12]) { return; }
	var sq = 0.0;
	for (var col = local; col < width; col += 256u) {
		let x = n_item(places[0], tokenStart + col);
		sq += x * x;
	}
${subgroups ? `	let sgSum = subgroupAdd(sq);
	if (sgLane == 0u) { modPartials[local / sgSize] = sgSum; }
	workgroupBarrier();
	var sqTotal = 0.0;
	let sgCount = (256u + sgSize - 1u) / sgSize;
	for (var i = 0u; i < sgCount; i++) { sqTotal += modPartials[i]; }
	let scale = inverseSqrt(sqTotal / f32(width) + 1e-6);` : `	modPartials[local] = sq;
	workgroupBarrier();
	for (var half = 128u; half > 0u; half /= 2u) {
		if (local < half) { modPartials[local] += modPartials[local + half]; }
		workgroupBarrier();
	}
	let scale = inverseSqrt(modPartials[0] / f32(width) + 1e-6);`}
	let shifts = places[7] * width;
	let scaling = shifts + width;
${packed ? `	for (var pair = local; pair < width / 2u; pair += 256u) {
		let col = pair * 2u;
		let x = n_pair(places[0], tokenStart + col);
		let w = n_pair(places[2], col);
		let mScale = n_pair(places[4], scaling + col);
		let mShift = n_pair(places[4], shifts + col);
		output[tokenStart / 2u + pair] = pack_bf16(
			x.x * scale * w.x * (1.0 + mScale.x) + mShift.x,
			x.y * scale * w.y * (1.0 + mScale.y) + mShift.y);
	}` : `	for (var col = local; col < width; col += 256u) {
		let x = n_item(places[0], tokenStart + col);
		let w = n_item(places[2], col);
		let mScale = n_item(places[4], scaling + col);
		let mShift = n_item(places[4], shifts + col);
		output[tokenStart + col] = x * scale * w * (1.0 + mScale) + mShift;
	}`}
}`;
for (const [key, sg] of [['strict_device_token_modulated', false], ['strict_device_token_modulated_sg', true]]) {
	const p17 = key + '_p17';
	Module['strictShaders'][key] = strictTokenModulated(sg, false);
	Module['strictShaders'][p17] = strictTokenModulated(sg, true);
	Module['strictPacked'][key] = p17;
	Module['strictCanPack'][key] = (places) => (places[3] & 1) === 0;
	Module['strictKernelGroups'][key] = Module['strictKernelGroups'][p17] = (places) => [places[12] / places[3], places[12] / places[3]];
}
Module['strictSubgroups']['strict_device_token_modulated'] = 'strict_device_token_modulated_sg';
// the numbers of the list at places[0] (places[1] of them) and then of the one at places[2] as 32 bit floats (lists of
// different formats joined on the GPU, strict_gpu_web_mixed): left, right
Module['strictShaders']['strict_joined'] = strictKernelBindings + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
	let at = id.x + id.y * groups.x * 64u;
	if (at >= places[4]) { return; }
	if (at < places[1]) {
		output[at] = n_item(places[0], at);
	} else {
		output[at] = n_item(places[2], at - places[1]);
	}
}`;
// a file a program writes in the browser (not the package loaded before it runs) is offered as a download on the page
// when it is closed, images are shown too; a program using no files has no FS
(Module['preRun'] ||= []).push(/** @suppress {undefinedVars} */ () => {
	if (typeof FS === 'undefined')
		return;
	const closed = FS.close;
	FS.close = (stream) => {
		const path = stream.path, isWritten = (stream.flags & 3) !== 0 && path && !path.startsWith('/dev/');
		const result = closed(stream);
		if (isWritten && runtimeInitialized && typeof document !== 'undefined') {
			const data = FS.readFile(path), writtenName = path.split('/').pop();
			const name = writtenName.endsWith('.part') ? writtenName.slice(0, -5) : writtenName;
			const link = document.createElement('a'), url = URL.createObjectURL(new Blob([data]));
			link.href = url;
			link.download = name;
			link.textContent = 'Download ' + name;
			document.body.append(link);
			if (/\.(png|jpe?g|webp|gif)$/i.test(name)) {
				const image = document.createElement('img');
				image.src = url;
				image.style.maxWidth = '100%';
				document.body.append(image);
			}
			console.log('Saved ' + name + ' (' + data.length + ' bytes), offered as a download');
		}
		return result;
	};
});
// files the page maps to URLs in Module.strictFiles that are read whole (any but .safetensors and .gguf, which are read
// in ranges) are in the file system of the program before it starts, folders of their names created: kept in the origin
// private file system after the first download, so File and Directory find them like files of the package
(Module['preRun'] ||= []).push(/** @suppress {undefinedVars} */ () => {
	const files = Module['strictFiles'] || {};
	const names = Object.keys(files).filter((name) => !/\.(safetensors|gguf)$/i.test(name));
	if (typeof FS === 'undefined' || !names.length)
		return;
	const kept = async (url) => {
		navigator['storage']['persist']?.().catch(() => {});
		const root = await navigator['storage']['getDirectory']();
		const folder = await (await root['getDirectoryHandle']('strict-files', { 'create': true }))
			['getDirectoryHandle'](encodeURIComponent(url), { 'create': true });
		const stored = await folder['getFileHandle']('whole').catch(() => null);
		if (stored)
			return new Uint8Array(await (await stored['getFile']())['arrayBuffer']());
		const response = await fetch(url, { cache: 'no-store' });
		if (!response.ok)
			throw url + ' answered ' + response.status;
		const bytes = new Uint8Array(await response.arrayBuffer());
		const writable = await (await folder['getFileHandle']('whole', { 'create': true }))['createWritable']();
		await writable['write'](bytes);
		await writable['close']();
		return bytes;
	};
	const fetched = (url) => kept(url).catch(() => fetch(url, { cache: 'no-store' }).then((response) => response.ok ?
		response.arrayBuffer() :
		Promise.reject(url + ' answered ' + response.status)).then((buffer) => new Uint8Array(buffer)));
	addRunDependency('strictFiles');
	Promise.all(names.map((name) => fetched(files[name]).then((bytes) => {
		const folder = name.substring(0, name.lastIndexOf('/'));
		if (folder)
			FS.mkdirTree(folder);
		FS.writeFile(name, bytes);
	}))).catch((error) => console.error('Reading a file of Module.strictFiles failed: ' + error))
		.finally(() => removeRunDependency('strictFiles'));
});
