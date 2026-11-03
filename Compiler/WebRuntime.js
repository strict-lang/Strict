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
		halves: {}, uploads: {}, views: [], borrowed: {}, lent: {}, dropped: {}, sixteens: {}, compiling: {},
		encoder: null, recorded: 0, fresh: 0, unused: [], staging: { current: null, full: [], free: [] }, pool: {},
		pooled: new WeakSet(), pooledBytes: 0, pass: null, prelude: null, flags: new WeakSet(), flagsUsed: [], flagPool: [],
		layout: null, lazy: {}, filed: {} },
	// kernels and copies are recorded into one command encoder, submitted before the host reads a result or after 64
	// dispatches (a submit per dispatch spent most of the time in Chrome's GPU process, Moss sampled 15 times slower than
	// natively); a kernel's numbers, places and the work of each dispatch get their own part of a staging buffer, which is
	// written again only after the GPU ran the commands of its submit, buffers are destroyed only after their submit
	// consecutive kernels share one compute pass (each pass is a command of its own), any other command ends it
	$strictWebGpuEncoder__deps: ['$strictWebGpu'],
	$strictWebGpuEncoder: () => {
		strictWebGpu.pass?.end();
		strictWebGpu.pass = null;
		return strictWebGpu.encoder ||= strictWebGpu.device.createCommandEncoder();
	},
	$strictWebGpuPass__deps: ['$strictWebGpu'],
	$strictWebGpuPass: () => strictWebGpu.pass ||=
		(strictWebGpu.encoder ||= strictWebGpu.device.createCommandEncoder()).beginComputePass(),
	// a failure flag of a kernel, cleared by a command submitted before its batch (not between its kernels, that would end
	// their pass): a flag given back during a batch is used again only from the next one on, so no kernel of the batch
	// that set it runs after the clear
	$strictWebGpuFlag__deps: ['$strictWebGpu'],
	$strictWebGpuFlag: () => {
		const reused = strictWebGpu.flagPool.pop();
		if (reused) {
			(strictWebGpu.prelude ||= strictWebGpu.device.createCommandEncoder()).clearBuffer(reused);
			return reused;
		}
		const flag = strictWebGpu.device.createBuffer({ size: 4, usage: 128 | 4 | 8 });
		strictWebGpu.flags.add(flag);
		return flag;
	},
	$strictWebGpuFlush__deps: ['$strictWebGpu'],
	$strictWebGpuFlush: () => {
		const encoder = strictWebGpu.encoder, staging = strictWebGpu.staging, device = strictWebGpu.device;
		if (!encoder)
			return;
		strictWebGpu.pass?.end();
		strictWebGpu.pass = null;
		strictWebGpu.encoder = null;
		strictWebGpu.recorded = 0;
		const used = staging.full.concat(staging.current || []), bytes = strictWebGpu.fresh;
		used.forEach((staged) => staged.used && device.queue.writeBuffer(staged.buffer, 0, staged.mirror, 0, staged.used));
		device.queue.submit(strictWebGpu.prelude ? [strictWebGpu.prelude.finish(), encoder.finish()] : [encoder.finish()]);
		strictWebGpu.prelude = null;
		strictWebGpu.flagPool.push(...strictWebGpu.flagsUsed);
		strictWebGpu.flagsUsed = [];
		strictWebGpu.unused.forEach((unused) => unused.destroy());
		strictWebGpu.unused = [];
		staging.full = [];
		staging.current = null;
		strictWebGpu.fresh = 0;
		strictWebGpu.queued += bytes;
		strictWebGpu.done.push(device.queue.onSubmittedWorkDone().then(() => {
			strictWebGpu.queued -= bytes;
			used.forEach((staged) => staging.free.push(Object.assign(staged, { used: 0 })));
			strictWebGpu.done.shift();
		}));
	},
	// a buffer of a kernel (strictWebGpuBuffer) goes back to its pool, the queue runs the commands in order, so the next
	// kernel writes it only after earlier ones read it (up to 2 GB are kept: 512 MB still left Moss creating 1100 buffers a
	// 480x848 step, 2 s instead of 1.5), others are destroyed now or, when commands
	// still to be submitted might use them, after their submit
	$strictWebGpuDestroyed__deps: ['$strictWebGpu'],
	$strictWebGpuDestroyed: (buffer) => {
		if (strictWebGpu.flags.has(buffer))
			(strictWebGpu.encoder ? strictWebGpu.flagsUsed : strictWebGpu.flagPool).push(buffer);
		else if (strictWebGpu.pooled.has(buffer) && strictWebGpu.pooledBytes + buffer.size <= 2147483648) {
			(strictWebGpu.pool[buffer.size + ' ' + buffer.usage] ||= []).push(buffer);
			strictWebGpu.pooledBytes += buffer.size;
		} else if (strictWebGpu.encoder)
			strictWebGpu.unused.push(buffer);
		else
			buffer.destroy();
	},
	$strictWebGpuBuffer__deps: ['$strictWebGpu'],
	$strictWebGpuBuffer: (bytes, usage) => {
		const size = Math.max(Math.ceil(bytes / 4) * 4, 4), free = strictWebGpu.pool[size + ' ' + usage];
		if (free && free.length) {
			strictWebGpu.pooledBytes -= size;
			return free.pop();
		}
		const buffer = strictWebGpu.device.createBuffer({ size, usage });
		strictWebGpu.pooled.add(buffer);
		return buffer;
	},
	// the numbers of a typed array at offset bytes into staged, copied into its staging buffer at the next submit
	$strictWebGpuStore__deps: ['$strictWebGpu'],
	$strictWebGpuStore: (staged, offset, numbers) => staged.mirror ?
		staged.mirror.set(new Uint8Array(numbers.buffer, numbers.byteOffset, numbers.byteLength), staged.offset + offset) :
		strictWebGpu.device.queue.writeBuffer(staged.buffer, staged.offset + offset, numbers),
	// bytes in a staging buffer of 64 MB (256 byte aligned for storage and uniform bindings) as a binding resource, more
	// in a buffer of their own
	$strictWebGpuStaged__deps: ['$strictWebGpu'],
	$strictWebGpuStaged: (bytes) => {
		const device = strictWebGpu.device, staging = strictWebGpu.staging, size = 67108864, usage = 128 | 64 | 8;
		const aligned = Math.ceil(Math.max(bytes, 4) / 256) * 256;
		if (aligned > size) {
			const own = device.createBuffer({ size: aligned, usage });
			strictWebGpu.unused.push(own);
			return { buffer: own, offset: 0, size: aligned, mirror: null };
		}
		if (!staging.current || staging.current.used + aligned > size) {
			if (staging.current)
				staging.full.push(staging.current);
			staging.current = staging.free.pop() ||
				{ buffer: device.createBuffer({ size, usage }), mirror: new Uint8Array(size), used: 0 };
		}
		const offset = staging.current.used;
		staging.current.used += aligned;
		return { buffer: staging.current.buffer, offset, size: aligned, mirror: staging.current.mirror };
	},
	// lines the page hands in with Module.strictInput(text) for Process.InputLine, which waits for the next one ("" when the
	// page has no more)
	$strictInput: { lines: [], current: '', waiting: null },
	$strictInput__postset: "Module['strictInput'] = (text) => { strictInput.lines.push(String(text)); " +
		"const waiting = strictInput.waiting; strictInput.waiting = null; if (waiting) waiting(); };",
	strict_webinput_length__deps: ['$strictInput', '$Asyncify', '$lengthBytesUTF8', 'strict_webfile_upload'],
	strict_webinput_length__async: true,
	strict_webinput_length: () => Asyncify.handleSleep((wakeUp) => {
		const taken = () => {
			strictInput.current = strictInput.lines.shift();
			wakeUp(lengthBytesUTF8(strictInput.current));
		};
		if (strictInput.lines.length)
			taken();
		else
			strictInput.waiting = taken;
	}),
	strict_webinput_copy__deps: ['$strictInput', '$stringToUTF8'],
	strict_webinput_copy: (target, size) => stringToUTF8(strictInput.current, target, size),
	// a file the page hands in with Module.strictFile(name, arrayBuffer) (an upload: a LoRA, a workflow) is written into
	// the file system of the program (folders of name created), which reads it like any other file; linked with
	// Process.InputLine, which tells the program about it
	strict_webfile_upload__deps: ['$FS'],
	strict_webfile_upload__postset: "Module['strictFile'] = (name, buffer) => _strict_webfile_upload(name, buffer);",
	strict_webfile_upload: (name, buffer) => {
		const path = String(name), folder = path.substring(0, path.lastIndexOf('/'));
		if (folder)
			FS.mkdirTree(folder);
		FS.writeFile(path, new Uint8Array(buffer));
	},
	strict_webgpu_ready__deps: ['$strictWebGpu', '$UTF8ToString'],
	strict_webgpu_ready: (name) => {
		const shaders = Module['strictShaders'], key = UTF8ToString(name);
		const usable = shaders && shaders[key] && strictWebGpu.pipelines[key] !== null;
		return usable && navigator.gpu && !strictWebGpu.failed ? 1 : 0;
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
			const isSixteen = isHalved && address in strictWebGpu.sixteens;
			places[slot * 2] = (isBound ? bound.length * 0x10000000 : total) + (isHalved ? 0x80000000 : 0) +
				(isSixteen ? 0x08000000 : 0);
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
	$strictWebGpuPipeline__deps: ['$strictWebGpu'],
	$strictWebGpuPipeline: async (device, code) => {
		const storage = (binding, type) => ({ binding, visibility: 4, buffer: { type, hasDynamicOffset: binding === 4 } });
		const types = ['storage', 'read-only-storage', 'read-only-storage', 'storage', 'uniform', 'read-only-storage',
			'read-only-storage', 'read-only-storage', 'read-only-storage'];
		// one layout for all kernels, the work of each dispatch of a kernel is a dynamic offset into its bind group
		const group = strictWebGpu.layout ||=
			device.createBindGroupLayout({ entries: types.map((type, binding) => storage(binding, type)) });
		const layout = device.createPipelineLayout({ bindGroupLayouts: [group] });
		device.pushErrorScope('validation');
		const module = device.createShaderModule({ code: Module['strictItem'] + code });
		const pipeline = device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
		const error = await device.popErrorScope();
		if (error)
			throw error.message;
		return pipeline;
	},
	// the device, every shader starts compiling on it right away (while a model loads): compiled on first use the
	// pipelines of Moss's first step took 2.5 s
	$strictWebGpuDevice__deps: ['$strictWebGpu', '$strictWebGpuPipeline'],
	$strictWebGpuDevice: async () => {
		const adapter = await navigator.gpu.requestAdapter();
		const limits = adapter && adapter.limits;
		const requiredLimits = limits && { maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
			maxBufferSize: limits.maxBufferSize,
			maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize };
		const device = adapter && await adapter.requestDevice({ requiredLimits });
		// any other error of WebGPU stops the program, results that were never computed must not be read
		device?.addEventListener('uncapturederror', (event) => abort('WebGPU error: ' + event.error.message));
		device?.lost.then((info) => abort('WebGPU device lost: ' + info.message));
		for (const key in device && Module['strictShaders'] || {}) {
			const compiled = strictWebGpuPipeline(device, Module['strictShaders'][key]);
			compiled.catch(() => {});
			strictWebGpu.compiling[key] = compiled;
		}
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
	$strictWebGpuRun__deps: ['$strictWebGpu', '$strictWebGpuPipeline', '$strictWebGpuDevice', '$strictWebGpuWritten',
		'$strictWebGpuEncoder', '$strictWebGpuFlush', '$strictWebGpuStaged', '$strictWebGpuDestroyed', '$strictWebGpuBuffer',
		'$strictWebGpuStore', '$strictWebGpuPass', '$strictWebGpuFlag'],
	$strictWebGpuRun: async (name, count, inputs, output, bits) => {
		const device = inputs && (strictWebGpu.device ||= await strictWebGpuDevice());
		if (inputs && !device) {
			console.warn('WebGPU has no adapter here, GPU loops and kernels run on the CPU');
			strictWebGpu.failed = true;
		}
		if (!device)
			return 0;
		// a kernel with a variant needing 32 KB of workgroup memory (Module.strictLarger) takes it where the device has it
		const larger = (Module['strictLarger'] || {})[name];
		if (larger && strictWebGpu.pipelines[larger] !== null && device.limits.maxComputeWorkgroupStorageSize >= 32768)
			try {
				strictWebGpu.pipelines[larger] ||= await (strictWebGpu.compiling[larger] ||
					strictWebGpuPipeline(device, Module['strictShaders'][larger]));
				name = larger;
			} catch {
				strictWebGpu.pipelines[larger] = null;
			}
		const pipeline = strictWebGpu.pipelines[name] ||= await (strictWebGpu.compiling[name] ||
			strictWebGpuPipeline(device, Module['strictShaders'][name]));
		const read = 128 | 4;
		const input = strictWebGpuStaged(inputs.size * 4), places = strictWebGpuStaged(inputs.places.byteLength);
		inputs.written.forEach(([offset, part]) => strictWebGpuStore(input, offset * 4, part));
		strictWebGpuStore(places, 0, inputs.places);
		const results = strictWebGpuBuffer(count * 4, read), failures = strictWebGpuFlag();
		const buffers = [{ buffer: results }, input, places, { buffer: failures }];
		const grouped = (Module['strictKernelGroups'] || {})[name], groups = Math.ceil(count / 64);
		const [tiles, part, splits = 1] = grouped ? grouped(inputs.places) : [groups, groups];
		const lists = [0, 1, 2, 3].map((at) => ({ buffer: (inputs.bound || [])[at] ||
			(strictWebGpu.empty ||= device.createBuffer({ size: 4, usage: 128 })) }));
		// a part of tiles is one dispatch, sized to stay short (Windows resets the device when a submit runs for 2 s)
		const dispatched = (used, tiles, part, bound) => {
			const parts = Math.ceil(tiles / part), work = strictWebGpuStaged(parts * 256);
			for (let at = 0; at < parts; at++)
				strictWebGpuStore(work, at * 256, new Uint32Array([at * part, splits]));
			const worked = { buffer: work.buffer, offset: work.offset, size: 8 };
			const entries = bound.concat(worked, lists).map((resource, binding) => ({ binding, resource }));
			const group = device.createBindGroup({ layout: strictWebGpu.layout, entries }), pass = strictWebGpuPass();
			pass.setPipeline(used);
			for (let at = 0; at < parts; at++) {
				const size = Math.min(part, tiles - at * part), wide = Math.min(size, 65535);
				pass.setBindGroup(0, group, [at * 256]);
				pass.dispatchWorkgroups(wide, Math.ceil(size / wide));
				strictWebGpu.recorded++;
			}
		};
		const runtimePipeline = async (key) => strictWebGpu.pipelines[key] ||=
			await (strictWebGpu.compiling[key] || strictWebGpuPipeline(device, Module['strictShaders'][key]));
		const splitting = splits > 1 && await runtimePipeline('strict_splits');
		const packing = !output && bits === 17 && await runtimePipeline('strict_packed');
		for (const [source, offset, length] of inputs.copies || [])
			strictWebGpuEncoder().copyBufferToBuffer(source, 0, input.buffer, input.offset + offset * 4, length * 4);
		strictWebGpu.fresh += count * 4;
		if (splitting) {
			const parts = strictWebGpuBuffer(count * splits * 4, 128);
			strictWebGpu.fresh += count * splits * 4;
			dispatched(pipeline, tiles, part, [{ buffer: parts }].concat(buffers.slice(1, 4)));
			dispatched(splitting, groups, groups, [buffers[0], { buffer: parts }, buffers[2], buffers[3]]);
			strictWebGpuDestroyed(parts);
		} else
			dispatched(pipeline, tiles, part, buffers);
		if (!output) {
			// a kept bfloat16 result is rounded like the CPU reads it and kept as halves, two in each 32 bits
			const halved = packing && strictWebGpuBuffer(Math.ceil(count / 2) * 4, read);
			if (halved) {
				const pairs = Math.ceil(count / 128);
				dispatched(packing, pairs, pairs, [{ buffer: halved }, buffers[0], buffers[2], buffers[3]]);
				strictWebGpuDestroyed(results);
			}
			const kept = halved || results;
			const handle = strictWebGpu.next++;
			strictWebGpu.buffers[handle] = kept;
			strictWebGpu.failures[handle] = failures;
			if (halved)
				strictWebGpu.halves[handle] = count;
			strictWebGpu.grown += kept.size;
			if (strictWebGpu.recorded >= 64)
				strictWebGpuFlush();
			// more than 1 GB of new results submitted ahead of the GPU waits for the oldest submit (2 GB or no limit made
			// 720x1280 Flux decodes vary from 3 to 6 s, 512 MB was slower)
			while (strictWebGpu.queued > 1073741824)
				await strictWebGpu.done[0];
			return handle;
		}
		const copy = strictWebGpuBuffer(count * 4 + 4, 1 | 8), encoder = strictWebGpuEncoder();
		encoder.copyBufferToBuffer(results, 0, copy, 0, count * 4);
		encoder.copyBufferToBuffer(failures, 0, copy, count * 4, 4);
		strictWebGpuFlush();
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
		[results, failures, copy].forEach(strictWebGpuDestroyed);
		return isFailed ? 0 : 1;
	},
	$strictWebGpuStarted__deps: ['$strictWebGpu', '$strictWebGpuRun', '$UTF8ToString', '$Asyncify'],
	$strictWebGpuStarted: (name, count, inputs, output, bits, placed) => Asyncify.handleSleep((wakeUp) => {
		const key = UTF8ToString(name);
		const running = (placed || Promise.resolve()).then(() => strictWebGpuRun(key, count, inputs(key), output, bits));
		running.then(wakeUp, (error) => {
			console.error('WebGPU kernel ' + key + ' runs on the CPU: ' + error);
			strictWebGpu.pipelines[key] = null;
			wakeUp(0);
		});
	}),
	strict_webgpu_kernel__deps: ['$strictWebGpuStarted', '$strictWebGpuKernelInputs', '$strictWebGpuPlacedInputs'],
	strict_webgpu_kernel__async: true,
	strict_webgpu_kernel: (name, count, table, slots, bits) => strictWebGpuStarted(name, count,
		() => strictWebGpuKernelInputs(count, table >>> 0, slots), 0, bits, strictWebGpuPlacedInputs(table >>> 0, slots)),
	// the numbers of a kept kernel result (strict_gpu_web_get) as bits into target: 1, 2 when its loop read outside of a
	// list, 0 when the buffer is gone
	strict_webgpu_download__deps: ['$strictWebGpu', '$strictWebGpuWritten', '$strictWebGpuWidened', '$strictWebGpuEncoder',
		'$strictWebGpuFlush', '$Asyncify', '$strictWebGpuBuffer', '$strictWebGpuDestroyed', '$strictWebGpuPlaced'],
	strict_webgpu_download__async: true,
	strict_webgpu_download: (handle, target, count, bits) => Asyncify.handleSleep((wakeUp) =>
		strictWebGpuPlaced(handle).then((source) => {
		if (!source)
			return wakeUp(0);
		const isHalved = handle in strictWebGpu.halves, bytes = isHalved ? Math.ceil(count / 2) * 4 : count * 4;
		const copy = strictWebGpuBuffer(bytes + 4, 1 | 8), encoder = strictWebGpuEncoder();
		encoder.copyBufferToBuffer(source, 0, copy, 0, bytes);
		encoder.copyBufferToBuffer(strictWebGpu.failures[handle], 0, copy, bytes, 4);
		strictWebGpuFlush();
		copy.mapAsync(1).then(() => {
			const mapped = copy.getMappedRange(), isOutside = new Uint32Array(mapped, bytes, 1)[0] !== 0;
			strictWebGpuWritten(isHalved ? strictWebGpuWidened(new Uint16Array(mapped, 0, count))
				: new Float32Array(mapped, 0, count), target >>> 0, bits);
			copy.unmap();
			strictWebGpuDestroyed(copy);
			wakeUp(isOutside ? 2 : 1);
		}, (error) => {
			console.error('WebGPU result could not be read: ' + error);
			wakeUp(0);
		});
	})),
	// a new kept result of the numbers of the left one and then of the right one (strict_gpu_web_joined), 0 on failure or
	// when the halves of both would not line up (the host then joins them)
	strict_webgpu_joined__deps: ['$strictWebGpu', '$strictWebGpuEncoder', '$strictWebGpuFlag'],
	strict_webgpu_joined: (left, right) => {
		const first = strictWebGpu.buffers[left], second = strictWebGpu.buffers[right], device = strictWebGpu.device;
		const firstHalves = strictWebGpu.halves[left], secondHalves = strictWebGpu.halves[right];
		if (!first || !second || (firstHalves === undefined) !== (secondHalves === undefined) ||
			(firstHalves !== undefined && (firstHalves === 0 || firstHalves % 2 !== 0)))
			return 0;
		const joined = device.createBuffer({ size: first.size + second.size, usage: 128 | 8 | 4 });
		const failures = strictWebGpuFlag();
		const encoder = strictWebGpuEncoder();
		encoder.copyBufferToBuffer(first, 0, joined, 0, first.size);
		encoder.copyBufferToBuffer(second, 0, joined, first.size, second.size);
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
	strict_webgpu_uploaded__deps: ['$strictWebGpu', '$strictWebGpuDevice', '$Asyncify', '$strictWebGpuFlag'],
	strict_webgpu_uploaded__async: true,
	strict_webgpu_uploaded: (address, words, halves) => Asyncify.handleSleep((wakeUp) => (async () => {
		const device = strictWebGpu.device ||= await strictWebGpuDevice();
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
			strictWebGpu.failures[handle] = strictWebGpuFlag();
			strictWebGpu.halves[handle] = words;
			strictWebGpu.uploads[handle] = words;
			strictWebGpu.grown += kept.size;
			return handle;
		}
		const isSixteen = halves === -2;
		halves = isSixteen ? words * 2 : Math.max(halves, 0);
		const uploaded = device.createBuffer({ size: Math.max(words * 4, 4), usage: 128 | 8 | 4 });
		device.queue.writeBuffer(uploaded, 0, HEAPU8, address >>> 0, words * 4);
		const handle = strictWebGpu.next++;
		strictWebGpu.buffers[handle] = uploaded;
		strictWebGpu.failures[handle] = strictWebGpuFlag();
		if (halves)
			strictWebGpu.halves[handle] = halves;
		strictWebGpu.uploads[handle] = halves || words;
		if (isSixteen)
			strictWebGpu.sixteens[handle] = true;
		strictWebGpu.grown += uploaded.size;
		return handle;
	})().then(wakeUp, (error) => {
		console.error('Uploading to WebGPU failed: ' + error);
		wakeUp(0);
	})),
	// the 16 MB blocks of files read with HTTP range requests (TensorFile reading a file that is not in the package),
	// the last 48 are kept until 5 s after the last read (fewer had Moss fetch a third of its file twice), with the
	// total size of each file from its first answer, the last block read of it and the bytes downloaded of it
	$strictWebFile: { blocks: new Map(), sizes: {}, lasts: {}, read: {}, folders: {}, timer: 0, writer: null,
		pending: new Map(), recent: new Map(), prefetched: {} },
	// the folder in the origin private file system keeping the downloaded blocks of the file at source (one file per
	// block and its size), null without one; the storage is asked to persist (else Chrome may evict it as best effort)
	$strictWebFileFolder__deps: ['$strictWebFile'],
	$strictWebFileFolder: (source) => strictWebFile.folders[source] ||= (async () => {
		navigator['storage']['persist']?.().catch(() => {});
		const root = await navigator['storage']['getDirectory']();
		const files = await root['getDirectoryHandle']('strict-files', { 'create': true });
		return await files['getDirectoryHandle'](encodeURIComponent(source), { 'create': true });
	})().catch(() => null),
	// the bytes of the file name in folder, null when it is not there
	$strictWebFileStored: async (folder, name) => {
		const handle = await folder['getFileHandle'](name).catch(() => null);
		return handle && new Uint8Array(await (await handle['getFile']())['arrayBuffer']());
	},
	// writes a copy of bytes as the file name in the folder of source, in a worker with a synchronous access handle
	// (writing from the page, which copies through a swap file, made the first load of Moss 20 instead of 6 s)
	$strictWebFileStore__deps: ['$strictWebFile'],
	$strictWebFileStore: (source, name, bytes) => {
		const writing = 'onmessage = async (event) => { const [folderName, name, bytes] = event.data; try { ' +
			'let folder = await (await navigator.storage.getDirectory()).getDirectoryHandle("strict-files", { create: true }); ' +
			'folder = await folder.getDirectoryHandle(folderName, { create: true }); ' +
			'const handle = await (await folder.getFileHandle(name, { create: true })).createSyncAccessHandle(); ' +
			'handle.truncate(0); handle.write(bytes, { at: 0 }); handle.flush(); handle.close(); } catch {} ' +
			'postMessage([folderName, name]); };';
		if (!strictWebFile.writer) {
			strictWebFile.writer = new Worker(URL.createObjectURL(new Blob([writing], { type: 'text/javascript' })));
			strictWebFile.writer.onmessage = (event) => strictWebFile.recent.delete(event.data[0] + ' ' + event.data[1]);
		}
		const copy = bytes.slice();
		strictWebFile.writer.postMessage([encodeURIComponent(source), name, copy], [copy.buffer]);
	},
	// downloads block index of the file at url (from source), keeps it in the origin private file system when the page
	// maps it (a block stays in recent until written) and prints the download progress; one request per block at a time
	$strictWebFileFetched__deps: ['$strictWebFile', '$strictWebFileStore'],
	$strictWebFileFetched: (url, source, index, isKept) => {
		const cache = strictWebFile, key = url + ' ' + index, pending = cache.pending.get(key);
		if (pending)
			return pending;
		const fetched = (async () => {
			const first = index * 16777216, range = { headers: { Range: 'bytes=' + first + '-' + (first + 16777215) },
				cache: isKept ? 'no-store' : 'default' };
			const response = await fetch(source, range).catch(() => fetch(source, range)).catch(() => fetch(source, range));
			if (response.status !== 206)
				throw source + ' answered ' + response.status + ' to a range request';
			cache.sizes[url] = Number((response.headers.get('Content-Range') || '').split('/')[1]) || cache.sizes[url];
			const bytes = new Uint8Array(await response.arrayBuffer());
			if (isKept) {
				if (!cache.read[url])
					strictWebFileStore(source, 'size', new TextEncoder().encode(String(cache.sizes[url])));
				cache.recent.set(encodeURIComponent(source) + ' ' + index, bytes);
				strictWebFileStore(source, String(index), bytes);
				const read = (cache.read[url] ||= new Map()).set(index, bytes.length);
				const got = [...read.values()].reduce((sum, length) => sum + length, 0);
				out('Downloading ' + url + ': ' + Math.round(got / 1048576) + ' of ' +
					Math.round(cache.sizes[url] / 1048576) + ' MB');
			}
			return bytes;
		})().finally(() => cache.pending.delete(key));
		cache.pending.set(key, fetched);
		return fetched;
	},
	// every block of a mapped file not kept yet is downloaded in the background, 8 at a time (one request at a time
	// gave 15, 8 gave 63 MB/s from Hugging Face, more failed), the first time one of its blocks has to be downloaded
	$strictWebFilePrefetch__deps: ['$strictWebFile', '$strictWebFileFetched'],
	$strictWebFilePrefetch: (url, source, folder) => {
		const cache = strictWebFile;
		if (cache.prefetched[url] || !cache.sizes[url])
			return;
		cache.prefetched[url] = true;
		const blocks = Math.ceil(cache.sizes[url] / 16777216);
		let next = 0;
		const kept = async (index) => {
			const handle = await folder['getFileHandle'](String(index)).catch(() => null);
			return handle && (await handle['getFile']())['size'] === Math.min(16777216, cache.sizes[url] - index * 16777216);
		};
		const worker = async () => {
			while (next < blocks) {
				const index = next++;
				if (!cache.pending.has(url + ' ' + index) && !cache.read[url]?.has(index) && !await kept(index))
					await strictWebFileFetched(url, source, index, true).catch(() => {});
			}
		};
		for (let count = 0; count < 8; count++)
			worker();
	},
	// the promise of the bytes of block index of the file at url, rejected when the server does not answer a range; a
	// request failing on the way is sent up to 3 times (bursts of parallel requests failed now and then); a file name
	// the page maps to a URL in Module.strictFiles ({ 'taef2.safetensors': 'https://huggingface.co/...' }) is read from
	// there (not into the HTTP cache of the browser, which kept a second copy of every block), its blocks are kept in the
	// origin private file system and read from it on the next visit (one of another
	// length than the file size says is read again), every block
	// downloaded prints "Downloading name: 16 of 98 MB" (bytes read so far, kept ones too) for the page's progress bar
	$strictWebFileBlock__deps: ['$strictWebFile', '$strictWebFileFolder', '$strictWebFileStored', '$strictWebFileFetched',
		'$strictWebFilePrefetch'],
	$strictWebFileBlock: (url, index) => {
		const cache = strictWebFile, key = url + ' ' + index;
		let block = cache.blocks.get(key);
		if (block)
			cache.blocks.delete(key);
		else {
			block = (async () => {
				const mapped = (Module['strictFiles'] || {})[url], source = mapped || url;
				const folder = mapped && await strictWebFileFolder(source);
				const recent = folder && cache.recent.get(encodeURIComponent(source) + ' ' + index);
				if (recent)
					return recent;
				const stored = folder && await strictWebFileStored(folder, String(index)).catch(() => null);
				const size = stored && !cache.sizes[url] && await strictWebFileStored(folder, 'size').catch(() => null);
				const total = cache.sizes[url] || (size ? Number(new TextDecoder().decode(size)) : 0);
				if (stored && stored.length === Math.min(16777216, total - index * 16777216)) {
					cache.sizes[url] ||= total;
					(cache.read[url] ||= new Map()).set(index, stored.length);
					return stored;
				}
				const bytes = await strictWebFileFetched(url, source, index, !!folder);
				if (folder)
					strictWebFilePrefetch(url, source, folder);
				return bytes;
			})();
			block.catch(() => cache.blocks.delete(key));
			if (cache.blocks.size >= 48)
				cache.blocks.delete(cache.blocks.keys().next().value);
		}
		cache.blocks.set(key, block);
		return block;
	},
	// bytes of the file at url from offset out of its blocks, each piece of a block given to place with its position
	// in the read, the 8 blocks after them are requested too when the read starts at or right after the one before (not
	// after a jump): Moss asks for one tensor after the other, which kept too few requests in flight (one stream gave
	// 200 MB/s, it loaded in 15 s), and its file holds layer 10 after 1
	$strictWebFileRange__deps: ['$strictWebFile', '$strictWebFileBlock'],
	$strictWebFileRange: async (url, offset, bytes, place) => {
		const size = 16777216, cache = strictWebFile, first = Math.floor(offset / size);
		const last = Math.floor((offset + bytes - 1) / size), blocks = [];
		for (let index = first; index <= last; index++)
			blocks.push(strictWebFileBlock(url, index));
		const isNear = first >= (cache.lasts[url] ?? -9) - 1 && first <= (cache.lasts[url] ?? -9) + 2;
		const ahead = isNear && !cache.prefetched[url] ? 8 : 0;
		cache.lasts[url] = last;
		for (let next = last + 1; next <= last + ahead && next * size < (cache.sizes[url] || 0); next++)
			strictWebFileBlock(url, next).catch(() => {});
		for (let index = first; index <= last; index++) {
			const block = await blocks[index - first], start = index * size;
			const from = Math.max(offset, start), to = Math.min(offset + bytes, start + block.length);
			if (to > from)
				place(block.subarray(from - start, to - start), from - offset);
		}
		clearTimeout(cache.timer);
		cache.timer = setTimeout(() => cache.blocks.clear(), 5000);
	},
	strict_webfile_mapped__deps: ['$UTF8ToString'],
	strict_webfile_mapped: (name) => (Module['strictFiles'] || {})[UTF8ToString(name)] ? 1 : 0,
	strict_webfile_read__deps: ['$strictWebFileRange', '$UTF8ToString', '$Asyncify'],
	strict_webfile_read__async: true,
	strict_webfile_read: (url, offset, bytes, target) => Asyncify.handleSleep((wakeUp) => {
		strictWebFileRange(UTF8ToString(url), offset, bytes, (piece, at) => HEAPU8.set(piece, (target >>> 0) + at)).then(() => {
			wakeUp(1);
		}, (error) => {
			console.error('Reading ' + UTF8ToString(url) + ' failed: ' + error);
			wakeUp(0);
		});
	}),
	// the 8 bit rows of a matrix at offset of the file at url straight into a new kept WebGPU buffer: the codes of the rows
	// as they are in the file (rowBytes a row, 4 bit codes stay two in a byte), written from the read blocks without a
	// copy, then the 32 bit scale of every row from scales (strict_eights reads that layout when a row is only its codes):
	// its handle, 0 on failure; the buffer is made only when a kernel, a download or a whole borrow first uses the rows as
	// they are (strictWebGpuPlaced), rows joined from several matrices are read into their joined buffer straight from the
	// file: the 9B encoder held 3.3 GB of rows and 2.3 GB of joined copies at once and Firefox ran out of memory
	strict_webfile_rows__deps: ['$strictWebGpu', '$strictWebGpuDevice', '$UTF8ToString', '$Asyncify', '$strictWebGpuFlag',
		'$strictWebGpuFilled'],
	strict_webfile_rows__async: true,
	strict_webfile_rows: (url, offset, rows, rowBytes, codesBytes, stride, columns, scales) => Asyncify.handleSleep((wakeUp) => {
		const path = UTF8ToString(url), copied = HEAPF32.slice(scales >>> 2, (scales >>> 2) + rows);
		(async () => {
			strictWebGpu.device ||= await strictWebGpuDevice();
			const handle = strictWebGpu.next++, codes = rows * rowBytes, file = { path, offset, rowBytes, scales: copied };
			const fill = async (buffer) => {
				await strictWebGpuFilled(buffer, 0, path, offset, codes);
				strictWebGpu.device.queue.writeBuffer(buffer, codes, copied);
			};
			strictWebGpu.filed[handle] = { size: codes + rows * 4, file, fill };
			strictWebGpu.lazy[handle] = Object.assign({}, strictWebGpu.filed[handle]);
			strictWebGpu.failures[handle] = strictWebGpuFlag();
			strictWebGpu.uploads[handle] = rows * stride / 4;
			return handle;
		})().then(wakeUp, (error) => {
			console.error('Reading the rows of ' + path + ' failed: ' + error);
			wakeUp(0);
		});
	}),
	// bytes of the file at path from offset into buffer at, written from the read blocks without a copy when aligned
	$strictWebGpuFilled__deps: ['$strictWebGpu', '$strictWebFileRange'],
	$strictWebGpuFilled: async (buffer, at, path, offset, bytes) => {
		const queue = strictWebGpu.device.queue, isAligned = offset % 4 === 0 && at % 4 === 0;
		const whole = isAligned ? null : new Uint8Array(Math.ceil(bytes / 4) * 4);
		await strictWebFileRange(path, offset, bytes, (piece, from) => isAligned ?
			queue.writeBuffer(buffer, at + from, piece) : whole.set(piece, from));
		if (whole)
			queue.writeBuffer(buffer, at, whole);
	},
	// the buffer of a kept handle: rows of a file (strict_webfile_rows), rows joined only from files and whole borrows of
	// them get it on their first use, once, again after GpuMemory.Evicted gave it back (strict_webgpu_evicted)
	$strictWebGpuPlaced__deps: ['$strictWebGpu'],
	$strictWebGpuPlaced: (handle) => {
		const lazy = strictWebGpu.lazy[handle];
		if (!lazy)
			return Promise.resolve(strictWebGpu.buffers[handle]);
		return lazy.placing ||= (async () => {
			const buffer = lazy.borrowOf ? await strictWebGpuPlaced(lazy.borrowOf)
				: strictWebGpu.device.createBuffer({ size: lazy.size, usage: 128 | 8 | 4 });
			if (!lazy.borrowOf)
				await lazy.fill(buffer);
			if (strictWebGpu.lazy[handle] !== lazy) {
				lazy.borrowOf || buffer.destroy();
				return undefined;
			}
			delete strictWebGpu.lazy[handle];
			strictWebGpu.buffers[handle] = buffer;
			strictWebGpu.grown += lazy.borrowOf ? 0 : buffer.size;
			return buffer;
		})();
	},
	// the rows of files a kernel's table binds as kept handles, uploaded before its inputs are read
	$strictWebGpuPlacedInputs__deps: ['$strictWebGpu', '$strictWebGpuPlaced'],
	$strictWebGpuPlacedInputs: (table, slots) => {
		const entries = HEAPF64.subarray(table / 8, table / 8 + slots * 3), waits = [];
		for (let slot = 0; slot < slots; slot++)
			if (entries[slot * 3 + 2] === 255 && entries[slot * 3] in strictWebGpu.lazy)
				waits.push(strictWebGpuPlaced(entries[slot * 3]));
		return Promise.all(waits);
	},
	// the views of count rows at address whose block is the WebGPU buffer handle (strict_rows_web): stride bytes a row, 4
	// bit codes when nibbles is 1, so joined rows can be gathered from it
	strict_webgpu_rows_views__deps: ['$strictWebGpu'],
	strict_webgpu_rows_views: (address, count, handle, stride, nibbles) =>
		strictWebGpu.views.push({ first: address >>> 0, last: (address >>> 0) + count * 64, handle, stride, nibbles }),
	// the rows whose view addresses are at views as a kept WebGPU buffer, its handle and the stride in words and nibbles
	// flag at shape, 0 when a row is in no buffer or rows differ: all rows of one block in order borrow its buffer (it is
	// destroyed when neither the rows nor any borrower is left), rows joined from several matrices are copied into a new
	// one on the GPU or read from their file when their block is not uploaded, one copy per run of rows that follow each
	// other in one block (Moss's query, key and value rows took 1.5 s row by row)
	strict_webgpu_gathered__deps: ['$strictWebGpu', '$strictWebGpuEncoder', '$strictWebGpuFlag', '$Asyncify',
		'$strictWebGpuPlaced', '$strictWebGpuFilled'],
	strict_webgpu_gathered__async: true,
	strict_webgpu_gathered: (views, count, shape) => Asyncify.handleSleep((wakeUp) => {
		const device = strictWebGpu.device, runs = [];
		let block = null;
		for (let row = 0; row < count; row++) {
			const address = HEAPU32[(views >>> 2) + row];
			if (!block || address < block.first || address >= block.last)
				block = strictWebGpu.views.findLast((placed) => address >= placed.first && address < placed.last);
			const first = runs.length ? runs[0].block : block, last = runs[runs.length - 1];
			if (!device || !block || block.stride !== first.stride || block.nibbles !== first.nibbles)
				return wakeUp(0);
			const index = (address - block.first) / 64;
			if (last && last.block === block && last.row + last.count === index)
				last.count++;
			else
				runs.push({ block, row: index, at: row, count: 1 });
		}
		const stride = runs[0].block.stride, source = runs[0].block;
		const isWhole = runs.length === 1 && runs[0].row === 0 && count === (source.last - source.first) / 64;
		const scalesOf = (block) => (block.last - block.first) / 64 * stride;
		const handle = strictWebGpu.next++;
		(async () => {
			const fromFile = async (joined, run, file) => {
				const from = file.offset + run.row * file.rowBytes;
				await strictWebGpuFilled(joined, run.at * stride, file.path, from, run.count * file.rowBytes);
				device.queue.writeBuffer(joined, count * stride + run.at * 4, file.scales, run.row, run.count);
			};
			const files = runs.map((run) => strictWebGpu.filed[run.block.handle]?.file);
			if (isWhole) {
				strictWebGpu.borrowed[handle] = source.handle;
				strictWebGpu.lent[source.handle] = (strictWebGpu.lent[source.handle] || 0) + 1;
				if (files[0]) {
					strictWebGpu.filed[handle] = { borrowOf: source.handle };
					strictWebGpu.lazy[handle] = { borrowOf: source.handle };
				} else
					strictWebGpu.buffers[handle] = strictWebGpu.buffers[source.handle];
			} else {
				const joined = device.createBuffer({ size: count * stride + count * 4, usage: 128 | 8 | 4 });
				if (files.every(Boolean))
					strictWebGpu.filed[handle] = { size: joined.size, fill: (buffer) =>
						Promise.all(runs.map((run, at) => fromFile(buffer, run, files[at]))) };
				await Promise.all(runs.map(async (run, at) => {
					if (files[at] && !strictWebGpu.buffers[run.block.handle])
						return fromFile(joined, run, files[at]);
					const buffer = strictWebGpu.buffers[run.block.handle], encoder = strictWebGpuEncoder();
					encoder.copyBufferToBuffer(buffer, run.row * stride, joined, run.at * stride, run.count * stride);
					encoder.copyBufferToBuffer(buffer, scalesOf(run.block) + run.row * 4, joined, count * stride + run.at * 4,
						run.count * 4);
				}));
				strictWebGpu.buffers[handle] = joined;
				strictWebGpu.grown += joined.size;
			}
			strictWebGpu.failures[handle] = strictWebGpuFlag();
			strictWebGpu.uploads[handle] = count * stride / 4;
			HEAP32[shape >>> 2] = stride / 4;
			HEAP32[(shape >>> 2) + 1] = source.nibbles;
			return handle;
		})().then(wakeUp, (error) => {
			console.error('Gathering rows on WebGPU failed: ' + error);
			wakeUp(0);
		});
	}),
	// a released block that borrowers still use keeps its buffer and file until the last of them is released
	strict_webgpu_release__deps: ['$strictWebGpu', '$strictWebGpuForgotten'],
	strict_webgpu_release: (handle) => {
		const lender = strictWebGpu.borrowed[handle], lent = strictWebGpu.lent, dropped = strictWebGpu.dropped;
		strictWebGpu.views = strictWebGpu.views.filter((placed) => placed.handle !== handle);
		if (!lender && lent[handle]) {
			dropped[handle] = true;
			return;
		}
		strictWebGpuForgotten(handle, !lender);
		if (lender && !--lent[lender] && dropped[lender]) {
			delete dropped[lender];
			strictWebGpuForgotten(lender, true);
		}
	},
	$strictWebGpuForgotten__deps: ['$strictWebGpu', '$strictWebGpuDestroyed'],
	$strictWebGpuForgotten: (handle, isOwner) => {
		if (isOwner && strictWebGpu.buffers[handle])
			strictWebGpuDestroyed(strictWebGpu.buffers[handle]);
		strictWebGpu.failures[handle] && strictWebGpuDestroyed(strictWebGpu.failures[handle]);
		for (const kept of [strictWebGpu.borrowed, strictWebGpu.failures, strictWebGpu.lazy, strictWebGpu.filed,
			strictWebGpu.buffers, strictWebGpu.halves, strictWebGpu.uploads, strictWebGpu.sixteens, strictWebGpu.lent])
			delete kept[handle];
	},
	// GpuMemory.Evicted gives back the buffers of rows read from files and of rows joined only from them, their next use
	// uploads them again from the file (kept in the origin private file system): MossWeb's Moss and the 9B encoder did not
	// fit next to each other in Firefox
	strict_webgpu_evicted__deps: ['$strictWebGpu', '$strictWebGpuFlush'],
	strict_webgpu_evicted: () => {
		strictWebGpuFlush();
		for (const handle in strictWebGpu.filed) {
			const buffer = strictWebGpu.buffers[handle], record = strictWebGpu.filed[handle];
			if (!buffer)
				continue;
			record.borrowOf || buffer.destroy();
			delete strictWebGpu.buffers[handle];
			strictWebGpu.lazy[handle] = Object.assign({}, record);
		}
	},
	// GpuMemory.Released and Evicted destroy the pooled buffers of finished kernels and the free failure flags after the
	// commands still to be submitted (Firefox ran out of memory loading Moss with the encoder's buffers pooled)
	strict_webgpu_trimmed__deps: ['$strictWebGpu', '$strictWebGpuFlush'],
	strict_webgpu_trimmed: () => {
		strictWebGpuFlush();
		Object.values(strictWebGpu.pool).flat().concat(strictWebGpu.flagPool).forEach((buffer) => buffer.destroy());
		strictWebGpu.pool = {};
		strictWebGpu.flagPool = [];
		strictWebGpu.pooledBytes = 0;
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
