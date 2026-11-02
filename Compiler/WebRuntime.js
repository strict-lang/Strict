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
		inputs: {}, halves: {}, uploads: {}, views: [], borrowed: {}, lent: {}, dropped: {}, sixteens: {},
		compiling: {} },
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
	// the device, every shader starts compiling on it right away (while a model loads): compiled on first use the
	// pipelines of Moss's first step took 2.5 s
	$strictWebGpuDevice__deps: ['$strictWebGpu', '$strictWebGpuPipeline'],
	$strictWebGpuDevice: async () => {
		const adapter = await navigator.gpu.requestAdapter();
		const limits = adapter && adapter.limits;
		const requiredLimits = limits && { maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
			maxBufferSize: limits.maxBufferSize };
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
	$strictWebGpuRun__deps: ['$strictWebGpu', '$strictWebGpuPipeline', '$strictWebGpuDevice', '$strictWebGpuWritten'],
	$strictWebGpuRun: async (name, count, inputs, output, bits) => {
		const device = inputs && (strictWebGpu.device ||= await strictWebGpuDevice());
		if (inputs && !device) {
			console.warn('WebGPU has no adapter here, GPU loops and kernels run on the CPU');
			strictWebGpu.failed = true;
		}
		if (!device)
			return 0;
		const pipeline = strictWebGpu.pipelines[name] ||= await (strictWebGpu.compiling[name] ||
			strictWebGpuPipeline(device, Module['strictShaders'][name]));
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
			await (strictWebGpu.compiling[key] || strictWebGpuPipeline(device, Module['strictShaders'][key]));
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
	strict_webgpu_uploaded__deps: ['$strictWebGpu', '$strictWebGpuDevice', '$Asyncify'],
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
			strictWebGpu.failures[handle] = device.createBuffer({ size: 4, usage: 128 | 8 | 4 });
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
		strictWebGpu.failures[handle] = device.createBuffer({ size: 4, usage: 128 | 8 | 4 });
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
	$strictWebFile: { blocks: new Map(), sizes: {}, lasts: {}, read: {}, folders: {}, timer: 0, writer: null },
	// the folder in the origin private file system keeping the downloaded blocks of the file at source (one file per
	// block and its size), null without one
	$strictWebFileFolder__deps: ['$strictWebFile'],
	$strictWebFileFolder: (source) => strictWebFile.folders[source] ||= (async () => {
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
		const writing = 'onmessage = async (event) => { try { const [folderName, name, bytes] = event.data; ' +
			'let folder = await (await navigator.storage.getDirectory()).getDirectoryHandle("strict-files", { create: true }); ' +
			'folder = await folder.getDirectoryHandle(folderName, { create: true }); ' +
			'const handle = await (await folder.getFileHandle(name, { create: true })).createSyncAccessHandle(); ' +
			'handle.truncate(0); handle.write(bytes, { at: 0 }); handle.flush(); handle.close(); } catch {} };';
		strictWebFile.writer ||= new Worker(URL.createObjectURL(new Blob([writing], { type: 'text/javascript' })));
		const copy = bytes.slice();
		strictWebFile.writer.postMessage([encodeURIComponent(source), name, copy], [copy.buffer]);
	},
	// the promise of the bytes of block index of the file at url, rejected when the server does not answer a range; a
	// request failing on the way is sent up to 3 times (bursts of parallel requests failed now and then); a file name
	// the page maps to a URL in Module.strictFiles ({ 'taef2.safetensors': 'https://huggingface.co/...' }) is read from
	// there, its blocks are kept in the origin private file system and read from it on the next visit (one of another
	// length than the file size says is read again), every block
	// downloaded prints "Downloading name: 16 of 98 MB" (bytes read so far, kept ones too) for the page's progress bar
	$strictWebFileBlock__deps: ['$strictWebFile', '$strictWebFileFolder', '$strictWebFileStored', '$strictWebFileStore'],
	$strictWebFileBlock: (url, index) => {
		const cache = strictWebFile, key = url + ' ' + index;
		let block = cache.blocks.get(key);
		if (block)
			cache.blocks.delete(key);
		else {
			block = (async () => {
				const mapped = (Module['strictFiles'] || {})[url], source = mapped || url;
				const folder = mapped && await strictWebFileFolder(source);
				const stored = folder && await strictWebFileStored(folder, String(index)).catch(() => null);
				const size = stored && !cache.sizes[url] && await strictWebFileStored(folder, 'size').catch(() => null);
				const total = cache.sizes[url] || (size ? Number(new TextDecoder().decode(size)) : 0);
				if (stored && stored.length === Math.min(16777216, total - index * 16777216)) {
					cache.sizes[url] ||= total;
					(cache.read[url] ||= new Map()).set(index, stored.length);
					return stored;
				}
				const first = index * 16777216, range = { headers: { Range: 'bytes=' + first + '-' + (first + 16777215) } };
				const response = await fetch(source, range).catch(() => fetch(source, range)).catch(() => fetch(source, range));
				if (response.status !== 206)
					throw source + ' answered ' + response.status + ' to a range request';
				cache.sizes[url] = Number((response.headers.get('Content-Range') || '').split('/')[1]) || cache.sizes[url];
				const bytes = new Uint8Array(await response.arrayBuffer());
				if (folder && !cache.read[url])
					strictWebFileStore(source, 'size', new TextEncoder().encode(String(cache.sizes[url])));
				if (folder)
					strictWebFileStore(source, String(index), bytes);
				if (mapped) {
					const read = (cache.read[url] ||= new Map()).set(index, bytes.length);
					const got = [...read.values()].reduce((sum, length) => sum + length, 0);
					out('Downloading ' + url + ': ' + Math.round(got / 1048576) + ' of ' +
						Math.round(cache.sizes[url] / 1048576) + ' MB');
				}
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
		const previous = cache.lasts[url] ?? -9, ahead = first >= previous - 1 && first <= previous + 2 ? 8 : 0;
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
	// its handle, 0 on failure
	strict_webfile_rows__deps: ['$strictWebGpu', '$strictWebGpuDevice', '$strictWebFileRange', '$UTF8ToString', '$Asyncify'],
	strict_webfile_rows__async: true,
	strict_webfile_rows: (url, offset, rows, rowBytes, codesBytes, stride, columns, scales) => Asyncify.handleSleep((wakeUp) => {
		const path = UTF8ToString(url);
		(async () => {
			const device = strictWebGpu.device ||= await strictWebGpuDevice();
			const codes = rows * rowBytes, uploaded = device.createBuffer({ size: codes + rows * 4, usage: 128 | 8 | 4 });
			const isAligned = offset % 4 === 0, whole = isAligned ? null : new Uint8Array(codes);
			await strictWebFileRange(path, offset, codes, (piece, at) => isAligned ?
				device.queue.writeBuffer(uploaded, at, piece) : whole.set(piece, at));
			if (whole)
				device.queue.writeBuffer(uploaded, 0, whole);
			device.queue.writeBuffer(uploaded, codes, HEAPF32, scales >>> 2, rows);
			const handle = strictWebGpu.next++;
			strictWebGpu.buffers[handle] = uploaded;
			strictWebGpu.grown += uploaded.size;
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
	// flag at shape, 0 when a row is in no buffer or rows differ: all rows of one block in order borrow its buffer (it is
	// destroyed when neither the rows nor any borrower is left), rows joined from several matrices are copied into a new
	// one on the GPU, one copy per run of rows that follow each other in one block (Moss's query, key and value rows took
	// 1.5 s row by row)
	strict_webgpu_gathered__deps: ['$strictWebGpu'],
	strict_webgpu_gathered: (views, count, shape) => {
		const device = strictWebGpu.device, runs = [];
		let block = null;
		for (let row = 0; row < count; row++) {
			const address = HEAPU32[(views >>> 2) + row];
			if (!block || address < block.first || address >= block.last)
				block = strictWebGpu.views.findLast((placed) => address >= placed.first && address < placed.last);
			const first = runs.length ? runs[0].block : block, last = runs[runs.length - 1];
			if (!device || !block || block.stride !== first.stride || block.nibbles !== first.nibbles)
				return 0;
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
		if (isWhole) {
			strictWebGpu.buffers[handle] = strictWebGpu.buffers[source.handle];
			strictWebGpu.borrowed[handle] = source.handle;
			strictWebGpu.lent[source.handle] = (strictWebGpu.lent[source.handle] || 0) + 1;
		} else {
			const joined = device.createBuffer({ size: count * stride + count * 4, usage: 128 | 8 | 4 });
			const encoder = device.createCommandEncoder();
			runs.forEach((run) => {
				const buffer = strictWebGpu.buffers[run.block.handle];
				encoder.copyBufferToBuffer(buffer, run.row * stride, joined, run.at * stride, run.count * stride);
				encoder.copyBufferToBuffer(buffer, scalesOf(run.block) + run.row * 4, joined, count * stride + run.at * 4,
					run.count * 4);
			});
			device.queue.submit([encoder.finish()]);
			strictWebGpu.buffers[handle] = joined;
			strictWebGpu.grown += joined.size;
		}
		strictWebGpu.failures[handle] = device.createBuffer({ size: 4, usage: 128 | 8 | 4 });
		strictWebGpu.uploads[handle] = count * stride / 4;
		HEAP32[shape >>> 2] = stride / 4;
		HEAP32[(shape >>> 2) + 1] = source.nibbles;
		return handle;
	},
	strict_webgpu_release__deps: ['$strictWebGpu'],
	strict_webgpu_release: (handle) => {
		const lender = strictWebGpu.borrowed[handle], lent = strictWebGpu.lent, dropped = strictWebGpu.dropped;
		if (lender && !--lent[lender] && dropped[lender]) {
			dropped[lender].destroy();
			delete dropped[lender];
		} else if (!lender && lent[handle])
			dropped[handle] = strictWebGpu.buffers[handle];
		else if (!lender)
			strictWebGpu.buffers[handle]?.destroy();
		strictWebGpu.views = strictWebGpu.views.filter((placed) => placed.handle !== handle);
		delete strictWebGpu.borrowed[handle];
		strictWebGpu.failures[handle]?.destroy();
		delete strictWebGpu.failures[handle];
		delete strictWebGpu.buffers[handle];
		delete strictWebGpu.halves[handle];
		delete strictWebGpu.uploads[handle];
		delete strictWebGpu.sixteens[handle];
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
