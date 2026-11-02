// emscripten_scan_registers continues in a setTimeout, the browser then shows a half drawn frame (black blinks on every
// collection). This scans right after Asyncify unwound the stack and rewinds in the same task.
addToLibrary({
	strict_scan_registers__deps: ['$Asyncify'],
	strict_scan_registers__async: true,
	strict_scan_registers: (func) => Asyncify.handleSleep((wakeUp) => {
		const stopUnwind = Asyncify.maybeStopUnwind;
		Asyncify.maybeStopUnwind = () => {
			const unwound = Asyncify.currData && Asyncify.state === Asyncify.State.Unwinding && !Asyncify.exportCallStack.length;
			stopUnwind();
			if (!unwound)
				return;
			Asyncify.maybeStopUnwind = stopUnwind;
			const stackBegin = Asyncify.currData + {{{ C_STRUCTS.asyncify_data_s.__size__ }}};
			const stackEnd = {{{ makeGetValue('Asyncify.currData', 0, '*') }}};
			{{{ makeDynCall('vpp', 'func') }}}(stackBegin, stackEnd);
			wakeUp();
		};
	}),
});
// A loop with a WGSL shader (WgslShader, strict_gpu_shaded) runs on WebGPU: its values go into one buffer of 32 bit
// floats (members with value.Bits is 32 are rounded to them), any other value that is no exact 32 bit float, a failed
// list read or a shader error runs the loop on the CPU.
// Closure knows no GPUBufferUsage: storage 128, copy destination 8, copy source 4, map read 1, compute stage 4.
addToLibrary({
	$strictWebGpu: { device: null, failed: false, pipelines: {}, buffers: {}, failures: {}, next: 1, grown: 0, queued: 0, done: [],
		inputs: {}, halves: {}, uploads: {}, views: [], borrowed: {} },
	strict_webgpu_ready__deps: ['$strictWebGpu', '$UTF8ToString'],
	strict_webgpu_ready: (name) => {
		const shaders = Module['strictShaders'], key = UTF8ToString(name);
		const usable = shaders && shaders[key] && strictWebGpu.pipelines[key] !== null;
		return usable && navigator.gpu && !strictWebGpu.failed ? 1 : 0;
	},
	$strictWebGpuInputs: (name, count, table, slots) => {
		const singles = Module['strictSingles'][name];
		const entries = HEAPF64.slice(table / 8, table / 8 + slots * 2);
		const places = new Uint32Array(slots * 2 + 1);
		let total = 0;
		for (let slot = 0; slot < slots; slot++) {
			places[slot * 2] = total;
			places[slot * 2 + 1] = entries[slot * 2 + 1] < 0 ? 1 : entries[slot * 2 + 1];
			total += places[slot * 2 + 1];
		}
		places[slots * 2] = count;
		const numbers = new Float32Array(Math.max(total, 1));
		for (let slot = 0; slot < slots; slot++) {
			const isList = entries[slot * 2 + 1] >= 0, source = isList ? entries[slot * 2] / 8 : 0;
			const isSingle = singles.includes(slot);
			for (let item = 0; item < places[slot * 2 + 1]; item++) {
				const value = isList ? HEAPF64[source + item] : entries[slot * 2];
				numbers[places[slot * 2] + item] = value;
				if (numbers[places[slot * 2] + item] !== value && value === value && !isSingle)
					return null;
			}
		}
		return { size: numbers.length, written: [[0, numbers]], places };
	},
	// a runtime kernel (NativeRuntimeWgsl.js) takes every value rounded to 32 bit floats (a whole number also exactly as
	// the second place of its slot): lists of 64 bit, 32 bit or bfloat16 (bits 17) numbers, other formats run on the CPU;
	// a kept kernel result (bits 255, its address the buffer) is bound as it is (the first 4, n_item) or copied into place
	// on the GPU (bfloat16 halves as they are, their place gets the top bit); only the other values are
	// written (the buffer of all of them starts as zeros), a 944 MB placeholder of a resident map took seconds
	$strictWebGpuKernelInputs__deps: ['$strictWebGpu', '$strictWebGpuWidened'],
	$strictWebGpuKernelInputs: (count, table, slots) => {
		const entries = HEAPF64.slice(table / 8, table / 8 + slots * 3);
		const places = new Uint32Array(slots * 2 + 1), parts = [], copies = [], bound = [];
		let total = 0;
		for (let slot = 0; slot < slots; slot++) {
			const address = entries[slot * 3], length = entries[slot * 3 + 1], bits = entries[slot * 3 + 2];
			const resident = bits === 255 && strictWebGpu.buffers[address];
			if (resident && address in strictWebGpu.uploads && strictWebGpu.uploads[address] !== length)
				return null;
			const isHalved = resident && address in strictWebGpu.halves, words = isHalved ? Math.ceil(length / 2) : length;
			const isBound = resident && bound.length < 4;
			if (isBound)
				bound.push(resident);
			else if (resident)
				copies.push([resident, total, words]);
			const part = length < 0 ? [address] : resident ? { length: isBound ? 0 : words, isResident: true }
				: bits === 64 ? HEAPF64.subarray(address / 8, address / 8 + length)
				: bits === 32 ? HEAPF32.subarray(address / 4, address / 4 + length)
				: bits === 17 ? strictWebGpuWidened(HEAPU16.subarray(address / 2, address / 2 + length)) : null;
			if (!part)
				return null;
			places[slot * 2] = (isBound ? bound.length * 0x10000000 : total) + (isHalved ? 0x80000000 : 0);
			places[slot * 2 + 1] = length < 0 ? (Number.isInteger(address) && address >= 0 ? address : 0) : resident ? length
				: part.length;
			parts.push(part);
			total += part.length;
		}
		places[slots * 2] = count;
		const written = [];
		parts.forEach((part, slot) => part.isResident || written.push([places[slot * 2],
			part instanceof Float32Array ? part : Float32Array.from(part)]));
		return { size: total, written, places, copies, bound };
	},
	// bfloat16 numbers as 32 bit floats
	$strictWebGpuWidened: (halves) => {
		const words = new Uint32Array(halves.length);
		for (let index = 0; index < halves.length; index++)
			words[index] = halves[index] << 16;
		return new Float32Array(words.buffer);
	},
	// a shader that does not compile rejects (its loop runs on the CPU), checked once per pipeline
	$strictWebGpuPipeline: async (device, code) => {
		const storage = (binding, type) => ({ binding, visibility: 4, buffer: { type } });
		const types = ['storage', 'read-only-storage', 'read-only-storage', 'storage', 'uniform', 'read-only-storage',
			'read-only-storage', 'read-only-storage', 'read-only-storage'];
		const group = device.createBindGroupLayout({ entries: types.map((type, binding) => storage(binding, type)) });
		const layout = device.createPipelineLayout({ bindGroupLayouts: [group] });
		device.pushErrorScope('validation');
		const module = device.createShaderModule({ code: Module['strictItem'] + code });
		const pipeline = device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
		const error = await device.popErrorScope();
		if (error)
			throw error.message;
		return pipeline;
	},
	$strictWebGpuDevice: async () => {
		const adapter = await navigator.gpu.requestAdapter();
		const limits = adapter && adapter.limits;
		const requiredLimits = limits && { maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
			maxBufferSize: limits.maxBufferSize };
		const device = adapter && await adapter.requestDevice({ requiredLimits });
		// any other error of WebGPU stops the program, results that were never computed must not be read
		device?.addEventListener('uncapturederror', (event) => abort('WebGPU error: ' + event.error.message));
		device?.lost.then((info) => abort('WebGPU device lost: ' + info.message));
		return device;
	},
	// numbers into the data at output as 64 bit, 32 bit or bfloat16 (bits 17, rounded to the nearest even) numbers
	$strictWebGpuWritten: (numbers, output, bits) => {
		if (bits === 64)
			HEAPF64.set(numbers, output / 8);
		else if (bits === 32)
			HEAPF32.set(numbers, output / 4);
		else {
			const words = new Uint32Array(numbers.buffer, numbers.byteOffset, numbers.length);
			const halves = HEAPU16.subarray(output / 2, output / 2 + numbers.length);
			for (let index = 0; index < words.length; index++)
				halves[index] = (words[index] + 0x7fff + ((words[index] >>> 16) & 1)) >>> 16;
		}
	},
	$strictWebGpuRun__deps: ['$strictWebGpu', '$strictWebGpuPipeline', '$strictWebGpuDevice', '$strictWebGpuWritten'],
	$strictWebGpuRun: async (name, count, inputs, output, bits) => {
		const device = inputs && (strictWebGpu.device ||= await strictWebGpuDevice());
		if (inputs && !device) {
			console.warn('WebGPU has no adapter here, GPU loops and kernels run on the CPU');
			strictWebGpu.failed = true;
		}
		if (!device)
			return 0;
		const pipeline = strictWebGpu.pipelines[name] ||= await strictWebGpuPipeline(device, Module['strictShaders'][name]);
		const written = 128 | 8, read = 128 | 4;
		const buffer = (size, flags) => device.createBuffer({ size: Math.max(size, 4), usage: flags });
		// one input buffer per power of two size is reused, the queue writes it only after earlier kernels read it (a new
		// one per kernel was cleared first and stayed in video memory until the GPU ran the kernel)
		const rounded = 2 ** Math.ceil(Math.log2(Math.max(inputs.size * 4, 4)));
		const input = rounded > device.limits.maxStorageBufferBindingSize ? buffer(inputs.size * 4, written)
			: strictWebGpu.inputs[rounded] ||= buffer(rounded, written);
		const isPooled = input === strictWebGpu.inputs[rounded];
		// places and work are reused like the inputs, the queue writes them only after earlier kernels read them
		if (!strictWebGpu.places || strictWebGpu.places.size < inputs.places.byteLength)
			strictWebGpu.places = buffer(Math.max(inputs.places.byteLength, 256), written);
		const work = strictWebGpu.work ||= buffer(16, 64 | 8);
		const buffers = [buffer(count * 4, read), input, strictWebGpu.places, buffer(4, read | 8)];
		const copy = output && buffer(count * 4 + 4, 1 | 8);
		inputs.written.forEach(([offset, part]) => device.queue.writeBuffer(buffers[1], offset * 4, part));
		device.queue.writeBuffer(buffers[2], 0, inputs.places);
		device.queue.writeBuffer(buffers[3], 0, new Uint32Array(1));
		const grouped = (Module['strictKernelGroups'] || {})[name], groups = Math.ceil(count / 64);
		const [tiles, part, splits = 1] = grouped ? grouped(inputs.places) : [groups, groups];
		// each submit is its own GPU packet, Windows resets the device when one runs for 2 s
		const lists = [0, 1, 2, 3].map((at) => (inputs.bound || [])[at] ||
			(strictWebGpu.empty ||= device.createBuffer({ size: 4, usage: 128 })));
		const dispatched = (used, tiles, part, bound = buffers.slice(0, 4)) => {
			for (let first = 0; first < tiles; first += part) {
				const size = Math.min(part, tiles - first), wide = Math.min(size, 65535);
				device.queue.writeBuffer(work, 0, new Uint32Array([first, splits]));
				const entries = bound.concat(work, lists).map((buffer, binding) => ({ binding, resource: { buffer } }));
				const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
				pass.setPipeline(used);
				pass.setBindGroup(0, device.createBindGroup({ layout: used.getBindGroupLayout(0), entries }));
				pass.dispatchWorkgroups(wide, Math.ceil(size / wide));
				pass.end();
				device.queue.submit([encoder.finish()]);
			}
		};
		const runtimePipeline = async (key) => strictWebGpu.pipelines[key] ||=
			await strictWebGpuPipeline(device, Module['strictShaders'][key]);
		if (inputs.copies && inputs.copies.length) {
			const copying = device.createCommandEncoder();
			for (const [source, offset, length] of inputs.copies)
				copying.copyBufferToBuffer(source, 0, buffers[1], offset * 4, length * 4);
			device.queue.submit([copying.finish()]);
		}
		if (splits > 1) {
			const parts = buffer(count * splits * 4, 128);
			dispatched(pipeline, tiles, part, [parts].concat(buffers.slice(1, 4)));
			dispatched(await runtimePipeline('strict_splits'), groups, groups, [buffers[0], parts, buffers[2], buffers[3]]);
			buffers.push(parts);
		} else
			dispatched(pipeline, tiles, part);
		// more than 1 GB of inputs and results queued ahead of the GPU waits for the oldest kernel (2 GB or no limit made
		// 720x1280 Flux decodes vary from 3 to 6 s, 512 MB was slower)
		const queued = (inputs.size + count) * 4;
		strictWebGpu.queued += queued;
		strictWebGpu.done.push(device.queue.onSubmittedWorkDone().then(() => {
			strictWebGpu.queued -= queued;
			strictWebGpu.done.shift();
		}));
		while (strictWebGpu.queued > 1073741824)
			await strictWebGpu.done[0];
		if (!output) {
			// a kept bfloat16 result is rounded like the CPU reads it and kept as halves, two in each 32 bits
			const halved = bits === 17 && buffer(Math.ceil(count / 2) * 4, read);
			if (halved) {
				const pairs = Math.ceil(count / 128);
				dispatched(await runtimePipeline('strict_packed'), pairs, pairs, [halved, buffers[0], buffers[2], buffers[3]]);
			}
			const kept = halved || buffers[0];
			buffers.forEach((used, at) => (at === 0 && !halved) || at === 2 || at === 3 || (at === 1 && isPooled) ||
				used.destroy());
			const handle = strictWebGpu.next++;
			strictWebGpu.buffers[handle] = kept;
			strictWebGpu.failures[handle] = buffers[3];
			if (halved)
				strictWebGpu.halves[handle] = count;
			strictWebGpu.grown += kept.size;
			return handle;
		}
		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(buffers[0], 0, copy, 0, count * 4);
		encoder.copyBufferToBuffer(buffers[3], 0, copy, count * 4, 4);
		device.queue.submit([encoder.finish()]);
		let isMapped = !navigator.userAgent.includes('Firefox');
		const mapping = copy.mapAsync(1).finally(() => isMapped = true);
		// Firefox maps only when it polls the device, every 100 ms or on a submit: empty submits make that 2 ms (Chrome
		// is 3 ms slower with them).
		while (!isMapped) {
			device.queue.submit([]);
			await new Promise((resolve) => {
				const channel = new MessageChannel();
				channel.port1.onmessage = resolve;
				channel.port2.postMessage(0);
			});
		}
		await mapping;
		const mapped = copy.getMappedRange(), isFailed = new Uint32Array(mapped, count * 4, 1)[0] !== 0;
		if (!isFailed)
			strictWebGpuWritten(new Float32Array(mapped, 0, count), output, bits);
		copy.unmap();
		buffers.concat(copy).forEach((used, at) => (at === 1 && isPooled) || at === 2 || used.destroy());
		return isFailed ? 0 : 1;
	},
	$strictWebGpuStarted__deps: ['$strictWebGpu', '$strictWebGpuRun', '$UTF8ToString', '$Asyncify'],
	$strictWebGpuStarted: (name, count, inputs, output, bits) => Asyncify.handleSleep((wakeUp) => {
		const key = UTF8ToString(name);
		strictWebGpuRun(key, count, inputs(key), output, bits).then(wakeUp, (error) => {
			console.error('WebGPU kernel ' + key + ' runs on the CPU: ' + error);
			strictWebGpu.pipelines[key] = null;
			wakeUp(0);
		});
	}),
	strict_webgpu__deps: ['$strictWebGpuStarted', '$strictWebGpuInputs'],
	strict_webgpu__async: true,
	strict_webgpu: (name, count, table, slots, output) =>
		strictWebGpuStarted(name, count, (key) => strictWebGpuInputs(key, count, table >>> 0, slots), output >>> 0, 64),
	strict_webgpu_kernel__deps: ['$strictWebGpuStarted', '$strictWebGpuKernelInputs'],
	strict_webgpu_kernel__async: true,
	strict_webgpu_kernel: (name, count, table, slots, bits) =>
		strictWebGpuStarted(name, count, () => strictWebGpuKernelInputs(count, table >>> 0, slots), 0, bits),
	// the numbers of a kept kernel result (strict_gpu_web_get) as bits into target: 1, 2 when its loop read outside of a
	// list, 0 when the buffer is gone
	strict_webgpu_download__deps: ['$strictWebGpu', '$strictWebGpuWritten', '$strictWebGpuWidened', '$Asyncify'],
	strict_webgpu_download__async: true,
	strict_webgpu_download: (handle, target, count, bits) => Asyncify.handleSleep((wakeUp) => {
		const source = strictWebGpu.buffers[handle], device = strictWebGpu.device;
		if (!source)
			return wakeUp(0);
		const isHalved = handle in strictWebGpu.halves, bytes = isHalved ? Math.ceil(count / 2) * 4 : count * 4;
		const copy = device.createBuffer({ size: bytes + 4, usage: 1 | 8 });
		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(source, 0, copy, 0, bytes);
		encoder.copyBufferToBuffer(strictWebGpu.failures[handle], 0, copy, bytes, 4);
		device.queue.submit([encoder.finish()]);
		copy.mapAsync(1).then(() => {
			const mapped = copy.getMappedRange(), isOutside = new Uint32Array(mapped, bytes, 1)[0] !== 0;
			strictWebGpuWritten(isHalved ? strictWebGpuWidened(new Uint16Array(mapped, 0, count))
				: new Float32Array(mapped, 0, count), target >>> 0, bits);
			copy.destroy();
			wakeUp(isOutside ? 2 : 1);
		}, (error) => {
			console.error('WebGPU result could not be read: ' + error);
			wakeUp(0);
		});
	}),
	// a new kept result of the numbers of the left one and then of the right one (strict_gpu_web_joined), 0 on failure or
	// when the halves of both would not line up (the host then joins them)
	strict_webgpu_joined__deps: ['$strictWebGpu'],
	strict_webgpu_joined: (left, right) => {
		const first = strictWebGpu.buffers[left], second = strictWebGpu.buffers[right], device = strictWebGpu.device;
		const firstHalves = strictWebGpu.halves[left], secondHalves = strictWebGpu.halves[right];
		if (!first || !second || (firstHalves === undefined) !== (secondHalves === undefined) ||
			(firstHalves !== undefined && (firstHalves === 0 || firstHalves % 2 !== 0)))
			return 0;
		const joined = device.createBuffer({ size: first.size + second.size, usage: 128 | 8 | 4 });
		const failures = device.createBuffer({ size: 4, usage: 128 | 8 | 4 });
		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(first, 0, joined, 0, first.size);
		encoder.copyBufferToBuffer(second, 0, joined, first.size, second.size);
		device.queue.submit([encoder.finish()]);
		strictWebGpu.buffers[strictWebGpu.next] = joined;
		strictWebGpu.failures[strictWebGpu.next] = failures;
		if (firstHalves !== undefined)
			strictWebGpu.halves[strictWebGpu.next] = firstHalves + secondHalves;
		strictWebGpu.grown += joined.size;
		return strictWebGpu.next++;
	},
	// a new kept buffer of the words at address as they are, of halves bfloat16 numbers when above 0, of 32 bit floats
	// kept as bfloat16 halves when -1 and every one is an exact bfloat16 number (Moss's BF16 weights read as 32 bit rows,
	// strict_gpu_web_uploaded), 0 before the first kernel made the device; a list that changed its length later runs its
	// kernel on the CPU
	strict_webgpu_uploaded__deps: ['$strictWebGpu'],
	strict_webgpu_uploaded: (address, words, halves) => {
		const device = strictWebGpu.device;
		if (!device)
			return 0;
		const floats = halves === -1 && HEAPU32.subarray(address >>> 2, (address >>> 2) + words);
		if (floats && floats.every((word) => (word & 0xffff) === 0)) {
			const narrowed = new Uint16Array(Math.ceil(words / 2) * 2);
			for (let at = 0; at < words; at++)
				narrowed[at] = floats[at] >>> 16;
			const kept = device.createBuffer({ size: narrowed.byteLength, usage: 128 | 8 | 4 });
			device.queue.writeBuffer(kept, 0, narrowed);
			const handle = strictWebGpu.next++;
			strictWebGpu.buffers[handle] = kept;
			strictWebGpu.failures[handle] = device.createBuffer({ size: 4, usage: 128 | 8 | 4 });
			strictWebGpu.halves[handle] = words;
			strictWebGpu.uploads[handle] = words;
			strictWebGpu.grown += kept.size;
			return handle;
		}
		halves = Math.max(halves, 0);
		const uploaded = device.createBuffer({ size: Math.max(words * 4, 4), usage: 128 | 8 | 4 });
		device.queue.writeBuffer(uploaded, 0, HEAPU8, address >>> 0, words * 4);
		const handle = strictWebGpu.next++;
		strictWebGpu.buffers[handle] = uploaded;
		strictWebGpu.failures[handle] = device.createBuffer({ size: 4, usage: 128 | 8 | 4 });
		if (halves)
			strictWebGpu.halves[handle] = halves;
		strictWebGpu.uploads[handle] = halves || words;
		strictWebGpu.grown += uploaded.size;
		return handle;
	},
	// bytes of the file at url from offset into target with an HTTP range request (TensorFile reading a file that is not
	// in the package), 1 or 0 when the server does not answer it as a range
	$strictWebFileRange: async (url, offset, bytes) => {
		const response = await fetch(url, { headers: { Range: 'bytes=' + offset + '-' + (offset + bytes - 1) } });
		if (response.status !== 206)
			throw url + ' answered ' + response.status + ' to a range request';
		return new Uint8Array(await response.arrayBuffer());
	},
	strict_webfile_read__deps: ['$strictWebFileRange', '$UTF8ToString', '$Asyncify'],
	strict_webfile_read__async: true,
	strict_webfile_read: (url, offset, bytes, target) => Asyncify.handleSleep((wakeUp) => {
		strictWebFileRange(UTF8ToString(url), offset, bytes).then((read) => {
			HEAPU8.set(read, target >>> 0);
			wakeUp(1);
		}, (error) => {
			console.error('Reading ' + UTF8ToString(url) + ' failed: ' + error);
			wakeUp(0);
		});
	}),
	// the 8 bit rows of a matrix at offset of the file at url straight into a new kept WebGPU buffer in the layout of
	// strict_rows (rowBytes codes a row in the file, each row at row * stride with its scale from scales for every 32 of
	// its columns after codesBytes, 4 bit codes stay two in a byte), read in parts of about 64 MB: its handle, 0 on failure
	strict_webfile_rows__deps: ['$strictWebGpu', '$strictWebGpuDevice', '$strictWebFileRange', '$UTF8ToString', '$Asyncify'],
	strict_webfile_rows__async: true,
	strict_webfile_rows: (url, offset, rows, rowBytes, codesBytes, stride, columns, scales) => Asyncify.handleSleep((wakeUp) => {
		const path = UTF8ToString(url);
		(async () => {
			const device = strictWebGpu.device ||= await strictWebGpuDevice();
			const uploaded = device.createBuffer({ size: Math.max(rows * stride, 4), usage: 128 | 8 | 4 });
			const parts = Math.ceil(columns / 32), chunk = Math.max(1, Math.floor(67108864 / rowBytes));
			for (let first = 0; first < rows; first += chunk) {
				const count = Math.min(chunk, rows - first);
				const read = await strictWebFileRange(path, offset + first * rowBytes, count * rowBytes);
				const placed = new Uint8Array(count * stride);
				for (let row = 0; row < count; row++) {
					placed.set(read.subarray(row * rowBytes, (row + 1) * rowBytes), row * stride);
					new Float32Array(placed.buffer, row * stride + codesBytes, parts).fill(HEAPF32[(scales >>> 2) + first + row]);
				}
				device.queue.writeBuffer(uploaded, first * stride, placed);
			}
			const handle = strictWebGpu.next++;
			strictWebGpu.buffers[handle] = uploaded;
			strictWebGpu.failures[handle] = device.createBuffer({ size: 4, usage: 128 | 8 | 4 });
			strictWebGpu.uploads[handle] = rows * stride / 4;
			return handle;
		})().then(wakeUp, (error) => {
			console.error('Reading the rows of ' + path + ' failed: ' + error);
			wakeUp(0);
		});
	}),
	// the views of count rows at address whose block is the WebGPU buffer handle (strict_rows_web): stride bytes a row, 4
	// bit codes when nibbles is 1, so joined rows can be gathered from it
	strict_webgpu_rows_views__deps: ['$strictWebGpu'],
	strict_webgpu_rows_views: (address, count, handle, stride, nibbles) =>
		strictWebGpu.views.push({ first: address >>> 0, last: (address >>> 0) + count * 64, handle, stride, nibbles }),
	// the rows whose view addresses are at views as a kept WebGPU buffer, its handle and the stride in words and nibbles
	// flag at shape, 0 when a row is in no buffer or rows differ: all rows of one block in order borrow its buffer (given
	// back without destroying it), rows joined from several matrices are copied into a new one on the GPU
	strict_webgpu_gathered__deps: ['$strictWebGpu'],
	strict_webgpu_gathered: (views, count, shape) => {
		const device = strictWebGpu.device, rows = [];
		for (let row = 0; row < count; row++) {
			const address = HEAPU32[(views >>> 2) + row];
			const block = strictWebGpu.views.findLast((placed) => address >= placed.first && address < placed.last);
			const first = rows.length ? rows[0].block : block;
			if (!device || !block || block.stride !== first.stride || block.nibbles !== first.nibbles)
				return 0;
			rows.push({ block, row: (address - block.first) / 64 });
		}
		const stride = rows[0].block.stride, source = rows[0].block;
		const isWhole = count === (source.last - source.first) / 64 && rows.every(({ block, row }, at) => block === source &&
			row === at);
		const handle = strictWebGpu.next++;
		if (isWhole) {
			strictWebGpu.buffers[handle] = strictWebGpu.buffers[source.handle];
			strictWebGpu.borrowed[handle] = true;
		} else {
			const joined = device.createBuffer({ size: count * stride, usage: 128 | 8 | 4 });
			const encoder = device.createCommandEncoder();
			rows.forEach(({ block, row }, at) => encoder.copyBufferToBuffer(strictWebGpu.buffers[block.handle], row * stride,
				joined, at * stride, stride));
			device.queue.submit([encoder.finish()]);
			strictWebGpu.buffers[handle] = joined;
			strictWebGpu.grown += joined.size;
		}
		strictWebGpu.failures[handle] = device.createBuffer({ size: 4, usage: 128 | 8 | 4 });
		strictWebGpu.uploads[handle] = count * stride / 4;
		HEAP32[shape >>> 2] = stride / 4;
		HEAP32[(shape >>> 2) + 1] = rows[0].block.nibbles;
		return handle;
	},
	strict_webgpu_release__deps: ['$strictWebGpu'],
	strict_webgpu_release: (handle) => {
		if (!strictWebGpu.borrowed[handle])
			strictWebGpu.buffers[handle]?.destroy();
		delete strictWebGpu.borrowed[handle];
		strictWebGpu.failures[handle]?.destroy();
		delete strictWebGpu.failures[handle];
		delete strictWebGpu.buffers[handle];
		delete strictWebGpu.halves[handle];
		delete strictWebGpu.uploads[handle];
	},
	// 1 once kept results took 1 GB since the last time, the collector then gives back the buffers of dead lists
	strict_webgpu_crowded__deps: ['$strictWebGpu'],
	strict_webgpu_crowded: () => {
		const isCrowded = strictWebGpu.grown > 1073741824;
		if (isCrowded)
			strictWebGpu.grown = 0;
		return isCrowded ? 1 : 0;
	},
});
