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
	$strictWebGpu: { device: null, failed: false, pipelines: {} },
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
		return { numbers, places };
	},
	// a runtime kernel (NativeRuntimeWgsl.js) takes every value rounded to 32 bit floats (a whole number also exactly as
	// the second place of its slot): lists of 64 bit, 32 bit or bfloat16 (bits 17) numbers, other formats run on the CPU
	$strictWebGpuKernelInputs: (count, table, slots) => {
		const entries = HEAPF64.slice(table / 8, table / 8 + slots * 3);
		const places = new Uint32Array(slots * 2 + 1), parts = [];
		let total = 0;
		for (let slot = 0; slot < slots; slot++) {
			const address = entries[slot * 3], length = entries[slot * 3 + 1], bits = entries[slot * 3 + 2];
			const part = length < 0 ? [address] : bits === 64 ? HEAPF64.subarray(address / 8, address / 8 + length)
				: bits === 32 ? HEAPF32.subarray(address / 4, address / 4 + length)
				: bits === 17 ? new Float32Array(Uint32Array.from(HEAPU16.subarray(address / 2, address / 2 + length),
					(half) => half << 16).buffer) : null;
			if (!part)
				return null;
			places[slot * 2] = total;
			places[slot * 2 + 1] = length < 0 ? (Number.isInteger(address) && address >= 0 ? address : 0) : part.length;
			parts.push(part);
			total += part.length;
		}
		places[slots * 2] = count;
		const numbers = new Float32Array(Math.max(total, 1));
		parts.forEach((part, slot) => numbers.set(part, places[slot * 2]));
		return { numbers, places };
	},
	$strictWebGpuPipeline: (device, code) => {
		const storage = (binding, type) => ({ binding, visibility: 4, buffer: { type } });
		const types = ['storage', 'read-only-storage', 'read-only-storage', 'storage', 'uniform'];
		const group = device.createBindGroupLayout({ entries: types.map((type, binding) => storage(binding, type)) });
		const layout = device.createPipelineLayout({ bindGroupLayouts: [group] });
		const module = device.createShaderModule({ code });
		return device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
	},
	$strictWebGpuDevice: async () => {
		const adapter = await navigator.gpu.requestAdapter();
		const limits = adapter && adapter.limits;
		const requiredLimits = limits && { maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
			maxBufferSize: limits.maxBufferSize };
		return adapter && await adapter.requestDevice({ requiredLimits });
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
		const buffers = [buffer(count * 4, read), buffer(inputs.numbers.byteLength, written),
			buffer(inputs.places.byteLength, written), buffer(4, read | 8)];
		const copy = buffer(count * 4 + 4, 1 | 8);
		device.queue.writeBuffer(buffers[1], 0, inputs.numbers);
		device.queue.writeBuffer(buffers[2], 0, inputs.places);
		device.queue.writeBuffer(buffers[3], 0, new Uint32Array(1));
		const grouped = (Module['strictKernelGroups'] || {})[name], groups = Math.ceil(count / 64);
		const [tiles, part] = grouped ? grouped(inputs.places) : [groups, groups];
		// each submit is its own GPU packet, Windows resets the device when one runs for 2 s
		for (let first = 0; first < tiles; first += part) {
			const firstGroup = buffer(16, 64 | 8), size = Math.min(part, tiles - first), wide = Math.min(size, 65535);
			device.queue.writeBuffer(firstGroup, 0, new Uint32Array([first]));
			buffers.push(firstGroup);
			const entries = buffers.slice(0, 4).concat(firstGroup).map((buffer, binding) => ({ binding, resource: { buffer } }));
			const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
			pass.setPipeline(pipeline);
			pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
			pass.dispatchWorkgroups(wide, Math.ceil(size / wide));
			pass.end();
			device.queue.submit([encoder.finish()]);
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
		buffers.concat(copy).forEach((used) => used.destroy());
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
	strict_webgpu_kernel: (name, count, table, slots, output, bits) =>
		strictWebGpuStarted(name, count, () => strictWebGpuKernelInputs(count, table >>> 0, slots), output >>> 0, bits),
});
