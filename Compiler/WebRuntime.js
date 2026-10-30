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
		const entries = HEAPF64.slice(table >> 3, (table >> 3) + slots * 2);
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
	$strictWebGpuPipeline: (device, code) => {
		const storage = (binding, type) => ({ binding, visibility: 4, buffer: { type } });
		const types = ['storage', 'read-only-storage', 'read-only-storage', 'storage'];
		const group = device.createBindGroupLayout({ entries: types.map((type, binding) => storage(binding, type)) });
		const layout = device.createPipelineLayout({ bindGroupLayouts: [group] });
		const module = device.createShaderModule({ code });
		return device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
	},
	$strictWebGpuDevice: async () => {
		const adapter = await navigator.gpu.requestAdapter();
		return adapter && await adapter.requestDevice();
	},
	$strictWebGpuRun__deps: ['$strictWebGpu', '$strictWebGpuInputs', '$strictWebGpuPipeline', '$strictWebGpuDevice'],
	$strictWebGpuRun: async (name, count, table, slots, output) => {
		const inputs = strictWebGpuInputs(name, count, table, slots);
		const device = inputs && (strictWebGpu.device ||= await strictWebGpuDevice());
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
		const entries = buffers.map((buffer, binding) => ({ binding, resource: { buffer } }));
		const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
		pass.setPipeline(pipeline);
		pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
		const groups = Math.ceil(count / 64), wide = Math.min(groups, 65535);
		pass.dispatchWorkgroups(wide, Math.ceil(groups / wide));
		pass.end();
		encoder.copyBufferToBuffer(buffers[0], 0, copy, 0, count * 4);
		encoder.copyBufferToBuffer(buffers[3], 0, copy, count * 4, 4);
		device.queue.submit([encoder.finish()]);
		await copy.mapAsync(1);
		const mapped = copy.getMappedRange(), isFailed = new Uint32Array(mapped, count * 4, 1)[0] !== 0;
		if (!isFailed)
			HEAPF64.set(new Float32Array(mapped, 0, count), output >> 3);
		copy.unmap();
		buffers.concat(copy).forEach((used) => used.destroy());
		return isFailed ? 0 : 1;
	},
	strict_webgpu__deps: ['$strictWebGpu', '$strictWebGpuRun', '$UTF8ToString', '$Asyncify'],
	strict_webgpu__async: true,
	strict_webgpu: (name, count, table, slots, output) => Asyncify.handleSleep((wakeUp) => {
		const key = UTF8ToString(name);
		strictWebGpuRun(key, count, table, slots, output).then(wakeUp, (error) => {
			console.error('WebGPU kernel ' + key + ' runs on the CPU: ' + error);
			strictWebGpu.pipelines[key] = null;
			wakeUp(0);
		});
	}),
});
