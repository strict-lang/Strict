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
	$strictWebGpu: { device: null, failed: false, destroying: false, pipelines: {}, buffers: {}, failures: {}, next: 1, grown: 0, queued: 0, done: [],
		halves: {}, uploads: {}, views: [], borrowed: {}, lent: {}, dropped: {}, sixteens: {}, compiling: {},
		encoder: null, empty: null, recorded: 0, fresh: 0, unused: [], staging: { current: null, full: [], free: [], size: 65536 }, pool: {},
		pooled: new WeakSet(), pooledBytes: 0, poolOrder: new Set(), pass: null, prelude: null, flags: new WeakSet(), flagsUsed: [], flagPool: [],
		layout: null, lazy: {}, filed: {}, profile: null,
		poolLimit: 0, queueLimit: 0 },
	$strictWebGpuAwaited__deps: ['$strictWebGpu'],
	$strictWebGpuAwaited: async (promise) => {
		if (!navigator.userAgent.includes('Firefox'))
			return await promise;
		let isDone = false, first = true;
		const watched = Promise.resolve(promise).finally(() => isDone = true);
		while (!isDone && strictWebGpu.device) {
			strictWebGpu.device.queue.submit([]);
			await new Promise((resolve) => {
				if (first) {
					first = false;
					const channel = new MessageChannel();
					channel.port1.onmessage = resolve;
					channel.port2.postMessage(0);
				} else
					setTimeout(resolve, 2);
			});
		}
		return await watched;
	},
	// kernels and copies are recorded into one command encoder, submitted before the host reads a result or after 64
	// dispatches (a submit per dispatch spent most of the time in Chrome's GPU process, Moss sampled 15 times slower than
	// natively); a kernel's numbers, places and the work of each dispatch get their own part of a staging buffer, which is
	// written again only after the GPU ran the commands of its submit, buffers are destroyed only after their submit
	// consecutive kernels share one compute pass (each pass is a command of its own), any other command ends it
	// phones and tablets, their GPU shares the memory with everything else
	$strictWebGpuMobile: () => /Android|iPhone|iPad|Mobi/.test(navigator.userAgent),
	// bytes ahead of the GPU (new results, buffers destroyed after their submit, staging buffers, uploads) a kernel waits
	// for the oldest submit above: 1 GB (2 GB or no limit made 720x1280 Flux decodes vary from 3 to 6 s, 512 MB was
	// slower), 256 MB on phones and tablets (a Galaxy S25 Ultra ran 3 GB ahead of its GPU, MossWeb's step 1 was killed)
	$strictWebGpuLimit__deps: ['$strictWebGpu', '$strictWebGpuMobile'],
	$strictWebGpuLimit: () => strictWebGpu.queueLimit ||= strictWebGpuMobile() ? 268435456 : 1073741824,
	// bytes written to the queue count until the GPU did them, then it waits while more than the limit is ahead of it
	$strictWebGpuThrottled__deps: ['$strictWebGpu', '$strictWebGpuLimit', '$strictWebGpuAwaited'],
	$strictWebGpuThrottled: async (bytes) => {
		if (bytes) {
			strictWebGpu.queued += bytes;
			strictWebGpu.done.push(strictWebGpu.device.queue.onSubmittedWorkDone().then(() => {
				strictWebGpu.queued -= bytes;
				strictWebGpu.done.shift();
			}));
		}
		while (strictWebGpu.queued > strictWebGpuLimit() && strictWebGpu.done.length)
			await strictWebGpuAwaited(strictWebGpu.done[0]);
	},
	$strictWebGpuEncoder__deps: ['$strictWebGpu'],
	$strictWebGpuEncoder: () => {
		strictWebGpu.pass?.end();
		strictWebGpu.pass = null;
		return strictWebGpu.encoder ||= strictWebGpu.device.createCommandEncoder();
	},
	$strictWebGpuPass__deps: ['$strictWebGpu', '$strictWebGpuCreated', '$strictWebGpuDestroyed'],
	$strictWebGpuPass: (label) => {
		const profile = strictWebGpu.profile;
		if (!profile)
			return strictWebGpu.pass ||= (strictWebGpu.encoder ||= strictWebGpu.device.createCommandEncoder()).beginComputePass();
		strictWebGpu.pass?.end();
		const slot = profile.entries.length;
		profile.entries.push({ label: label || 'unknown', slot });
		const querySet = profile.queries ||= strictWebGpu.device.createQuerySet({ type: 'timestamp', count: 256 });
		return strictWebGpu.pass = (strictWebGpu.encoder ||= strictWebGpu.device.createCommandEncoder()).beginComputePass({
			timestampWrites: { querySet, beginningOfPassWriteIndex: slot * 2, endOfPassWriteIndex: slot * 2 + 1 }
		});
	},
	// a failure flag of a kernel, cleared by a command submitted before its batch (not between its kernels, that would end
	// their pass): a flag given back during a batch is used again only from the next one on, so no kernel of the batch
	// that set it runs after the clear
	$strictWebGpuFlag__deps: ['$strictWebGpu', '$strictWebGpuCreated'],
	$strictWebGpuFlag: () => {
		const reused = strictWebGpu.flagPool.pop();
		if (reused) {
			(strictWebGpu.prelude ||= strictWebGpu.device.createCommandEncoder()).clearBuffer(reused);
			return reused;
		}
		const flag = strictWebGpuCreated(4, 128 | 4 | 8);
		strictWebGpu.flags.add(flag);
		return flag;
	},
	$strictWebGpuFlush__deps: ['$strictWebGpu', '$strictWebGpuMobile'],
	$strictWebGpuFlush: () => {
		const encoder = strictWebGpu.encoder, staging = strictWebGpu.staging, device = strictWebGpu.device;
		if (!encoder)
			return;
		strictWebGpu.pass?.end();
		strictWebGpu.pass = null;
		strictWebGpu.encoder = null;
		strictWebGpu.recorded = 0;
		const profile = strictWebGpu.profile;
		let profileCopy = null, profileEntries = null;
		if (profile && profile.entries.length) {
			profileEntries = profile.entries;
			profile.entries = [];
			const qBytes = profileEntries.length * 16;
			const resolve = profile.resolve ||= strictWebGpuCreated(4096, 512 | 4);
			profileCopy = strictWebGpuCreated(qBytes, 1 | 8);
			encoder.resolveQuerySet(profile.queries, 0, profileEntries.length * 2, resolve, 0);
			encoder.copyBufferToBuffer(resolve, 0, profileCopy, 0, qBytes);
		}
		const used = staging.full.concat(staging.current || []), destroyed = strictWebGpu.unused;
		const stagedBytes = used.reduce((sum, staged) => sum + (strictWebGpuMobile() ? staged.mirror.length : staged.used), 0);
		const bytes = strictWebGpu.fresh + destroyed.reduce((sum, buffer) => sum + buffer.size, 0) + stagedBytes;
		used.forEach((staged) => staged.used && device.queue.writeBuffer(staged.buffer, 0, staged.mirror, 0, staged.used));
		device.queue.submit(strictWebGpu.prelude ? [strictWebGpu.prelude.finish(), encoder.finish()] : [encoder.finish()]);
		strictWebGpu.prelude = null;
		strictWebGpu.flagPool.push(...strictWebGpu.flagsUsed);
		strictWebGpu.flagsUsed = [];
		destroyed.forEach((unused) => unused.destroy());
		strictWebGpu.unused = [];
		staging.full = [];
		staging.current = null;
		strictWebGpu.fresh = 0;
		strictWebGpu.queued += bytes;
		strictWebGpu.done.push(device.queue.onSubmittedWorkDone().then(async () => {
			strictWebGpu.queued -= bytes;
			used.forEach((staged) => strictWebGpuMobile() && staging.free.length
				? staged.buffer.destroy() : staging.free.push(Object.assign(staged, { used: 0 })));
			if (profileCopy) {
				await profileCopy.mapAsync(1);
				const times = new BigUint64Array(profileCopy.getMappedRange());
				for (const entry of profileEntries) {
					const dt = Number(times[entry.slot * 2 + 1] - times[entry.slot * 2]) / 1e6;
					const cur = profile.totals[entry.label] ||= { count: 0, ms: 0 };
					cur.count++;
					cur.ms += dt;
				}
				profileCopy.unmap();
				profileCopy.destroy();
				clearTimeout(profile.timer);
				profile.timer = setTimeout(() => Module['strictWebGpuProfile']?.(), 250);
			}
			strictWebGpu.done.shift();
		}));
	},
	// a buffer of a kernel (strictWebGpuBuffer) goes back to its pool, the queue runs the commands in order, so the next
	// kernel writes it only after earlier ones read it (up to 4 GB are kept: 2 GB still left Moss creating 1250 to 1700
	// buffers an image at 480x848, 3 s instead of 2; 128 MB on phones and tablets, whose GPU shares the memory: a Galaxy S25 Ultra
	// killed the page at 5.7 GB), others are destroyed now or, when commands still to be submitted might use them, after
	// their submit
	$strictWebGpuDestroyed__deps: ['$strictWebGpu', '$strictWebGpuMobile', '$strictWebGpuDropped'],
	$strictWebGpuDestroyed: (buffer) => {
		if (strictWebGpu.flags.has(buffer))
			(strictWebGpu.encoder ? strictWebGpu.flagsUsed : strictWebGpu.flagPool).push(buffer);
		else if (strictWebGpu.pooled.has(buffer) && buffer.size <= (strictWebGpu.poolLimit ||=
			strictWebGpuMobile() ? 134217728 : 2147483648)) {
			// the longest unused buffers make room: after another image size they are all of sizes no kernel asks for anymore
			const expire = strictWebGpu.next - 8000;
			for (const oldest of strictWebGpu.poolOrder) {
				if (strictWebGpu.pooledBytes + buffer.size <= strictWebGpu.poolLimit && !(oldest.__tick < expire))
					break;
				const free = strictWebGpu.pool[oldest.size + ' ' + oldest.usage];
				free.splice(free.indexOf(oldest), 1);
				strictWebGpu.poolOrder.delete(oldest);
				strictWebGpu.pooledBytes -= oldest.size;
				strictWebGpuDropped(oldest);
			}
			buffer.__tick = strictWebGpu.next;
			(strictWebGpu.pool[buffer.size + ' ' + buffer.usage] ||= []).push(buffer);
			strictWebGpu.poolOrder.add(buffer);
			strictWebGpu.pooledBytes += buffer.size;
		} else
			strictWebGpuDropped(buffer);
	},
	$strictWebGpuDropped__deps: ['$strictWebGpu'],
	$strictWebGpuDropped: (buffer) => {
		if (strictWebGpu.encoder)
			strictWebGpu.unused.push(buffer);
		else
			buffer.destroy();
	},
	$strictWebGpuTrim__deps: ['$strictWebGpu', '$strictWebGpuFlush', '$strictWebFile'],
	$strictWebGpuTrim: () => {
		strictWebGpuFlush();
		Object.values(strictWebGpu.pool).flat().concat(strictWebGpu.flagPool,
			strictWebGpu.staging.free.splice(0).map((staged) => staged.buffer)).forEach((buffer) => buffer.destroy());
		strictWebGpu.pool = {};
		strictWebGpu.poolOrder.clear();
		strictWebGpu.flagPool = [];
		strictWebGpu.pooledBytes = 0;
		strictWebGpu.grown = 0;
		strictWebFile.blocks.clear();
	},
	$strictWebGpuRelease__deps: ['$strictWebGpu', '$strictWebGpuTrim'],
	$strictWebGpuRelease__postset: "if (typeof window !== 'undefined') { window.addEventListener('pagehide', strictWebGpuRelease); window.addEventListener('beforeunload', strictWebGpuRelease); }",
	$strictWebGpuRelease: () => {
		if (!strictWebGpu.device && !strictWebGpu.empty)
			return;
		strictWebGpu.destroying = true;
		strictWebGpuTrim();
		const active = [strictWebGpu.empty, strictWebGpu.staging.current?.buffer,
			...strictWebGpu.staging.full.map((staged) => staged.buffer), ...strictWebGpu.flagsUsed, ...strictWebGpu.unused];
		for (const handle in strictWebGpu.buffers)
			if (!strictWebGpu.borrowed[handle])
				active.push(strictWebGpu.buffers[handle]);
		for (const handle in strictWebGpu.failures)
			active.push(strictWebGpu.failures[handle]);
		active.forEach((buffer) => buffer?.destroy());
		strictWebGpu.device?.destroy();
		strictWebGpu.device = null;
		strictWebGpu.empty = null;
		strictWebGpu.staging.current = null;
		strictWebGpu.staging.full = [];
		strictWebGpu.staging.size = 65536;
		strictWebGpu.flagsUsed = [];
		strictWebGpu.unused = [];
		strictWebGpu.buffers = {};
		strictWebGpu.failures = {};
		strictWebGpu.pipelines = {};
		strictWebGpu.compiling = {};
		strictWebGpu.precompiled = false;
		strictWebGpu.layout = null;
		strictWebGpu.done = [];
		strictWebGpu.queued = 0;
		strictWebGpu.lazy = {};
		strictWebGpu.filed = {};
		strictWebGpu.views = [];
		strictWebGpu.borrowed = {};
		strictWebGpu.lent = {};
		strictWebGpu.dropped = {};
		strictWebGpu.halves = {};
		strictWebGpu.uploads = {};
		strictWebGpu.sixteens = {};
	},
	$strictWebGpuOutOfMemory__deps: ['$strictWebGpu', '$strictWebGpuRelease'],
	$strictWebGpuOutOfMemory: (error) => {
		const text = String(error?.message || error || '');
		if (!/memory/i.test(text))
			return false;
		if (strictWebGpu.failed)
			return true;
		strictWebGpu.failed = true;
		strictWebGpuRelease();
		const message = 'WebGPU ran out of memory (' + text + '). Please close other browser tabs and reload.';
		console.error(message);
		if (typeof document !== 'undefined') {
			const status = document.getElementById('status'), overlay = document.getElementById('overlay');
			if (status)
				status.textContent = message;
			if (overlay)
				overlay.style.display = 'flex';
			else if (!status && document.body) {
				const banner = document.createElement('div');
				banner.style.cssText = 'position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:9999;background:#2b1116;color:#ffd7dc;border:1px solid #e88388;padding:10px 16px;border-radius:6px;font:14px system-ui,sans-serif';
				banner.textContent = message;
				document.body.appendChild(banner);
			}
		}
		return true;
	},
	$strictWebGpuCreated__deps: ['$strictWebGpu', '$strictWebGpuTrim'],
	$strictWebGpuCreated: (size, usage) => {
		try {
			return strictWebGpu.device.createBuffer({ size, usage });
		} catch {
			strictWebGpuTrim();
			return strictWebGpu.device.createBuffer({ size, usage });
		}
	},
	$strictWebGpuBuffer__deps: ['$strictWebGpu', '$strictWebGpuCreated'],
	$strictWebGpuBuffer: (bytes, usage) => {
		const size = Math.max(Math.ceil(bytes / 4) * 4, 4), free = strictWebGpu.pool[size + ' ' + usage];
		if (free && free.length) {
			strictWebGpu.pooledBytes -= size;
			const reused = free.pop();
			strictWebGpu.poolOrder.delete(reused);
			return reused;
		}
		const buffer = strictWebGpuCreated(size, usage);
		strictWebGpu.pooled.add(buffer);
		return buffer;
	},
	// the numbers of a typed array at offset bytes into staged, copied into its staging buffer at the next submit
	$strictWebGpuStore__deps: ['$strictWebGpu'],
	$strictWebGpuStore: (staged, offset, numbers) => staged.mirror ?
		staged.mirror.set(new Uint8Array(numbers.buffer, numbers.byteOffset, numbers.byteLength), staged.offset + offset) :
		strictWebGpu.device.queue.writeBuffer(staged.buffer, staged.offset + offset, numbers),
	// bytes in a staging buffer growing from 64 KB up to 64 MB (256 byte aligned for bindings), larger in its own buffer
	$strictWebGpuStaged__deps: ['$strictWebGpu', '$strictWebGpuCreated'],
	$strictWebGpuStaged: (bytes) => {
		const staging = strictWebGpu.staging, limit = 67108864, usage = 128 | 64 | 8;
		const aligned = Math.ceil(Math.max(bytes, 4) / 256) * 256;
		if (aligned > limit) {
			const own = strictWebGpuCreated(aligned, usage);
			strictWebGpu.unused.push(own);
			return { buffer: own, offset: 0, size: aligned, mirror: null };
		}
		if (!staging.current || staging.current.used + aligned > staging.current.mirror.length) {
			const overflowed = Boolean(staging.current);
			if (overflowed)
				staging.full.push(staging.current);
			const reuseIndex = staging.free.findIndex((entry) => entry.mirror.length >= aligned);
			if (reuseIndex >= 0)
				staging.current = staging.free.splice(reuseIndex, 1)[0];
			else {
				if (overflowed)
					staging.size = Math.min(limit, staging.size * 2);
				const size = Math.min(limit, Math.max(staging.size || 65536, aligned));
				staging.current = { buffer: strictWebGpuCreated(size, usage), mirror: new Uint8Array(size), used: 0 };
			}
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
	strict_webgpu_ready__deps: ['$strictWebGpu', '$UTF8ToString', '$strictWebGpuRelease'],
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
	$strictWebGpuKernelInputs__deps: ['$strictWebGpu', '$strictWebGpuWidened', '$strictWebGpuFloat8Widened'],
	$strictWebGpuKernelInputs: (count, table, slots) => {
		const decode = (addr, len, b) => b === 64 ? HEAPF64.subarray(addr / 8, addr / 8 + len)
			: b === 32 ? HEAPF32.subarray(addr / 4, addr / 4 + len)
			: b === 17 ? strictWebGpuWidened(HEAPU16.subarray(addr / 2, addr / 2 + len))
			: b === 9 ? HEAPU8.subarray(addr >>> 0, (addr >>> 0) + len)
			: b === 7 ? strictWebGpuFloat8Widened(HEAPU8.subarray(addr >>> 0, (addr >>> 0) + len)) : null;
		const entries = HEAPF64.slice(table / 8, table / 8 + slots * 3);
		const places = new Uint32Array(slots * 2 + 1), written = [], copies = [], bound = [];
		let total = 0;
		const pushHost = (part) => {
			const at = total;
			written.push([at, part instanceof Float32Array ? part : Float32Array.from(part)]);
			total += part.length;
			return at;
		};
		for (let slot = 0; slot < slots; slot++) {
			const address = entries[slot * 3], length = entries[slot * 3 + 1], bits = entries[slot * 3 + 2];
			if (bits === 65) {
				const descBase = address >>> 2, desc = new Uint32Array(length * 2), descStart = total;
				places[slot * 2] = descStart;
				places[slot * 2 + 1] = length;
				total += length * 2;
				for (let row = 0; row < length; row++) {
					const rAddr = HEAPU32[descBase + row * 3], rLen = HEAPU32[descBase + row * 3 + 1];
					const rBits = HEAPU32[descBase + row * 3 + 2];
					const rRes = rBits === 255 && strictWebGpu.buffers[rAddr];
					if (rBits === 255 && !rRes) return null;
					const rHalf = rRes && rAddr in strictWebGpu.halves, rWords = rHalf ? Math.ceil(rLen / 2) : rLen;
					const rBound = rRes && bound.length < 4;
					let rPlace;
					if (rBound) { bound.push(rRes); rPlace = bound.length * 0x10000000; }
					else if (rRes) { copies.push([rRes, total, rWords]); rPlace = total; total += rWords; }
					else {
						const rPart = decode(rAddr, rLen, rBits);
						if (!rPart) return null;
						rPlace = pushHost(rPart);
					}
					if (rHalf) rPlace += 0x80000000 + (rAddr in strictWebGpu.sixteens ? 0x08000000 : 0);
					desc[row * 2] = rPlace >>> 0;
					desc[row * 2 + 1] = rLen;
				}
				written.push([descStart, new Float32Array(desc.buffer)]);
				continue;
			}
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
				: decode(address, length, bits);
			if (!part)
				return null;
			const isSixteen = isHalved && address in strictWebGpu.sixteens;
			const offset = part.isResident ? total : pushHost(part);
			if (part.isResident) total += part.length;
			places[slot * 2] = (isBound ? bound.length * 0x10000000 : offset) + (isHalved ? 0x80000000 : 0) +
				(isSixteen ? 0x08000000 : 0);
			places[slot * 2 + 1] = length < 0 ? (Number.isInteger(address) && address >= 0 ? address : 0) : resident ? length
				: part.length;
		}
		places[slots * 2] = count;
		return { size: total, written, places, copies, bound };
	},
	// bfloat16 numbers as 32 bit floats
	$strictWebGpuWidened: (halves) => {
		const words = new Uint32Array(halves.length);
		for (let index = 0; index < halves.length; index++)
			words[index] = halves[index] << 16;
		return new Float32Array(words.buffer);
	},
	$strictWebGpuFloat8Widened: (bytes) => {
		const out = new Float32Array(bytes.length), words = new Uint32Array(out.buffer);
		for (let index = 0; index < bytes.length; index++) {
			const b = bytes[index], e = (b >> 3) & 15, m = b & 7, s = (b & 128) << 24;
			if (e === 0) out[index] = (b & 128 ? -m : m) * 0.001953125;
			else words[index] = s | ((e + 120) << 23) | (m << 20);
		}
		return out;
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
		const enabled = code.match(/^(enable\s+[\w-]+;\s*)+/);
		const prefix = enabled ? enabled[0] : '';
		if (prefix.includes('f16') && !device.features.has('shader-f16'))
			throw 'shader-f16 unavailable';
		if (prefix.includes('subgroups') && !device.features.has('subgroups'))
			throw 'subgroups unavailable';
		const body = prefix + Module['strictItem'] + code.slice(prefix.length);
		const module = device.createShaderModule({ code: body });
		const pipeline = device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'main' } });
		const error = await device.popErrorScope();
		if (error)
			throw error.message;
		return pipeline;
	},
	// compiles one kernel pipeline on demand (shared across variants and precompiled runs)
	$strictWebGpuCompile__deps: ['$strictWebGpu', '$strictWebGpuPipeline'],
	$strictWebGpuCompile: (device, key) => strictWebGpu.pipelines[key] ||
		(strictWebGpu.compiling[key] ||= strictWebGpuPipeline(device, Module['strictShaders'][key]))
			.then((pipeline) => (strictWebGpu.pipelines[key] = pipeline)),
	// neural network kernels start compiling while model rows stream in, not at device startup (36 unused shaders
	// delayed FractalZoomer's first frame by >22 s on Firefox's serial pipeline compiler)
	$strictWebGpuPrecompile__deps: ['$strictWebGpu', '$strictWebGpuPipeline'],
	$strictWebGpuPrecompile: (device) => {
		if (!device || strictWebGpu.precompiled)
			return;
		strictWebGpu.precompiled = true;
		for (const key in Module['strictShaders'] || {})
			if (key.startsWith('strict_')) {
				const compiled = strictWebGpu.compiling[key] ||=
					strictWebGpuPipeline(device, Module['strictShaders'][key]);
				compiled.catch(() => {});
			}
	},
	$strictWebGpuDevice__deps: ['$strictWebGpu', '$strictWebGpuRelease', '$strictWebGpuOutOfMemory'],
	$strictWebGpuDevice: async () => {
		if (strictWebGpu.failed)
			return null;
		strictWebGpu.destroying = false;
		try {
			const adapter = await navigator.gpu.requestAdapter();
			const limits = adapter && adapter.limits;
			const requiredLimits = limits && { maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
				maxBufferSize: limits.maxBufferSize,
				maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize };
			const wantProfile = typeof location !== 'undefined' && location.search.includes('profile') || Module['strictProfile'];
			const requiredFeatures = ['shader-f16', 'subgroups'].concat(wantProfile ? ['timestamp-query'] : [])
				.filter((feature) => adapter && adapter.features.has(feature));
			const device = adapter && await adapter.requestDevice({ requiredFeatures, requiredLimits });
			strictWebGpu.profile = device && device.features.has('timestamp-query')
				? { totals: {}, entries: [], queries: null, resolve: null } : null;
			if (strictWebGpu.profile) {
				const dump = (reset) => {
					const rows = Object.entries(strictWebGpu.profile.totals).sort((a, b) => b[1].ms - a[1].ms);
					const total = rows.reduce((sum, [, value]) => sum + value.ms, 0);
					const lines = rows.map(([key, value]) =>
						`${value.ms.toFixed(2).padStart(8)} ms (${String(value.count).padStart(5)}x, ${(value.ms * 100 / (total || 1)).toFixed(1).padStart(5)}%) ${key}`);
					console.log('[strict-webgpu-profile] total ' + total.toFixed(2) + ' ms\n' + lines.join('\n'));
					if (reset)
						strictWebGpu.profile.totals = {};
					return rows;
				};
				Module['strictWebGpuProfile'] = dump;
				if (typeof window !== 'undefined')
					window['strictWebGpuProfile'] = dump;
				if (typeof globalThis !== 'undefined')
					globalThis['strictWebGpuProfile'] = dump;
			}
			// any other error of WebGPU stops the program, results that were never computed must not be read
			device?.addEventListener('uncapturederror', (event) => strictWebGpuOutOfMemory(event.error) ||
				abort('WebGPU error: ' + event.error.message));
			device?.lost.then((info) => strictWebGpu.destroying || info.reason === 'destroyed' ||
				abort('WebGPU device lost: ' + info.message));
			return device;
		} catch (error) {
			if (!strictWebGpuOutOfMemory(error))
				throw error;
			return null;
		}
	},
	// numbers into the data at output as 64 bit, 32 bit, 8 bit (9 bytes, 7 E4M3) or bfloat16 (bits 17) numbers
	$strictWebGpuWritten: (numbers, output, bits) => {
		if (bits === 64)
			HEAPF64.set(numbers, output / 8);
		else if (bits === 32)
			HEAPF32.set(numbers, output / 4);
		else if (bits === 9) {
			const bytes = HEAPU8.subarray(output >>> 0, (output >>> 0) + numbers.length);
			for (let index = 0; index < numbers.length; index++)
				bytes[index] = Math.max(0, Math.min(255, Math.round(numbers[index])));
		} else if (bits === 7) {
			const bytes = HEAPU8.subarray(output >>> 0, (output >>> 0) + numbers.length);
			const words = new Uint32Array(numbers.buffer, numbers.byteOffset, numbers.length);
			for (let index = 0; index < numbers.length; index++) {
				const v = numbers[index], mag = Math.min(Math.abs(v), 448.0), sign = v < 0 ? 128 : 0;
				if (mag < 0.015625) bytes[index] = sign | Math.round(mag * 512.0);
				else {
					const b = words[index] & 0x7fffffff, f = b & 0x7fffff;
					const c = Math.min(126, (((b >>> 23) - 120) << 3) + ((f + 0x7ffff + ((f >>> 20) & 1)) >>> 20));
					bytes[index] = sign | c;
				}
			}
		} else {
			const words = new Uint32Array(numbers.buffer, numbers.byteOffset, numbers.length);
			const halves = HEAPU16.subarray(output / 2, output / 2 + numbers.length);
			for (let index = 0; index < words.length; index++)
				halves[index] = (words[index] + 0x7fff + ((words[index] >>> 16) & 1)) >>> 16;
		}
	},
	// records one kernel whose pipelines are already compiled. undefined when a pipeline is still missing, so the caller
	// takes the path that can wait; no await, a step is hundreds of these and unwinding the wasm stack for each one was
	// the whole 1.4 s
	$strictWebGpuPackedSource: (code) => {
		const entry = '@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {\nlet at = id.x + id.y * groups.x * 64u;\n';
		const pos = code.indexOf(entry);
		if (pos < 0)
			return null;
		const rest = code.slice(pos + entry.length);
		const bound = /^if \(at >= (places\[\d+\])\) \{ return; \}\n/.exec(rest);
		if (!bound)
			return null;
		const bodyAndEnd = rest.slice(bound[0].length);
		const outMatch = /\noutput\[at\] = ([^;\n]+);\n\}$/.exec(bodyAndEnd);
		if (!outMatch)
			return null;
		const body = bodyAndEnd.slice(0, outMatch.index);
		if (body.includes('output['))
			return null;
		return code.slice(0, pos) +
			'fn strict_eval(at: u32) -> f32 {\n' + body + '\nreturn ' + outMatch[1] + ';\n}\n' +
			'@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {\n' +
			'let at = id.x + id.y * groups.x * 64u;\nlet count = ' + bound[1] + ';\nif (at * 2u >= count) { return; }\n' +
			'let second = select(0.0, strict_eval(at * 2u + 1u), at * 2u + 1u < count);\n' +
			'output[at] = pack_bf16(strict_eval(at * 2u), second);\n}';
	},
	$strictWebGpuRecord__deps: ['$strictWebGpu', '$strictWebGpuEncoder', '$strictWebGpuFlush', '$strictWebGpuStaged',
		'$strictWebGpuDestroyed', '$strictWebGpuBuffer', '$strictWebGpuStore', '$strictWebGpuPass', '$strictWebGpuFlag',
		'$strictWebGpuLimit', '$strictWebGpuCreated'],
	$strictWebGpuRecord: (pipeline, name, count, inputs, output, bits) => {
		if (strictWebGpu.profile && strictWebGpu.profile.entries.length >= 124)
			strictWebGpuFlush();
		const device = strictWebGpu.device;
		const grouped = (Module['strictKernelGroups'] || {})[name], groups = Math.ceil(count / 64);
		let [tiles, part, splits = 1] = grouped ? grouped(inputs.places) : [groups, groups];
		const packKey = !output && bits === 17 && (Module['strictPacked'] || {})[name];
		const canPack = packKey && splits <= 1 && strictWebGpu.pipelines[packKey] &&
			(!(Module['strictCanPack'] || {})[name] || Module['strictCanPack'][name](inputs.places, count));
		if (splits > 1 && !strictWebGpu.pipelines['strict_splits'])
			return undefined;
		if (!output && bits === 17 && !canPack && !strictWebGpu.pipelines['strict_packed'])
			return undefined;
		if (canPack && !grouped)
			tiles = part = Math.ceil(count / 128);
		const read = 128 | 4, resultBytes = canPack ? Math.ceil(count / 2) * 4 : count * 4;
		const input = strictWebGpuStaged(inputs.size * 4), places = strictWebGpuStaged(inputs.places.byteLength);
		inputs.written.forEach(([offset, numbers]) => strictWebGpuStore(input, offset * 4, numbers));
		strictWebGpuStore(places, 0, inputs.places);
		const results = strictWebGpuBuffer(resultBytes, read), failures = strictWebGpuFlag();
		const buffers = [{ buffer: results }, input, places, { buffer: failures }];
		const lists = [0, 1, 2, 3].map((at) => ({ buffer: (inputs.bound || [])[at] ||
			(strictWebGpu.empty ||= strictWebGpuCreated(4, 128)) }));
		// a part of tiles is one dispatch, sized to stay short (Windows resets the device when a submit runs for 2 s)
		const dispatched = (used, tiles, part, bound, label) => {
			const parts = Math.ceil(tiles / part), work = strictWebGpuStaged(parts * 256);
			const workWords = strictWebGpu.workWords ||= new Uint32Array(2);
			for (let at = 0; at < parts; at++) {
				workWords[0] = at * part;
				workWords[1] = splits;
				strictWebGpuStore(work, at * 256, workWords);
			}
			const worked = { buffer: work.buffer, offset: work.offset, size: 8 };
			const entries = bound.concat(worked, lists).map((resource, binding) => ({ binding, resource }));
			const group = device.createBindGroup({ layout: strictWebGpu.layout, entries }), pass = strictWebGpuPass(label);
			pass.setPipeline(used);
			for (let at = 0; at < parts; at++) {
				const size = Math.min(part, tiles - at * part), wide = Math.min(size, 65535);
				pass.setBindGroup(0, group, [at * 256]);
				pass.dispatchWorkgroups(wide, Math.ceil(size / wide));
				strictWebGpu.recorded++;
			}
		};
		const splitting = splits > 1 && strictWebGpu.pipelines['strict_splits'];
		const packing = !output && bits === 17 && !canPack && strictWebGpu.pipelines['strict_packed'];
		for (const [source, offset, length] of inputs.copies || [])
			strictWebGpuEncoder().copyBufferToBuffer(source, 0, input.buffer, input.offset + offset * 4, length * 4);
		strictWebGpu.fresh += resultBytes;
		if (splitting) {
			const parts = strictWebGpuBuffer(count * splits * 4, 128);
			strictWebGpu.fresh += count * splits * 4;
			dispatched(pipeline, tiles, part, [{ buffer: parts }].concat(buffers.slice(1, 4)), name);
			dispatched(splitting, groups, groups, [buffers[0], { buffer: parts }, buffers[2], buffers[3]], 'strict_splits');
			strictWebGpuDestroyed(parts);
		} else {
			const usedName = canPack ? packKey : name;
			dispatched(canPack ? strictWebGpu.pipelines[packKey] : pipeline, tiles, part, buffers, usedName);
		}
		if (!output) {
			// a kept bfloat16 result is rounded like the CPU reads it and kept as halves, two in each 32 bits
			const halved = packing ? strictWebGpuBuffer(Math.ceil(count / 2) * 4, read) : canPack ? results : false;
			if (packing) {
				const pairs = Math.ceil(count / 128);
				dispatched(packing, pairs, pairs, [{ buffer: halved }, buffers[0], buffers[2], buffers[3]], 'strict_packed');
				strictWebGpuDestroyed(results);
			}
			const kept = halved || results;
			const handle = strictWebGpu.next++;
			strictWebGpu.buffers[handle] = kept;
			strictWebGpu.failures[handle] = failures;
			if (halved)
				strictWebGpu.halves[handle] = count;
			strictWebGpu.grown += kept.size;
			if (strictWebGpu.recorded >= 64 || strictWebGpu.fresh > strictWebGpuLimit() / 4)
				strictWebGpuFlush();
			return handle;
		}
		const copy = strictWebGpuBuffer(count * 4 + 4, 1 | 8), encoder = strictWebGpuEncoder();
		encoder.copyBufferToBuffer(results, 0, copy, 0, count * 4);
		encoder.copyBufferToBuffer(failures, 0, copy, count * 4, 4);
		strictWebGpuFlush();
		return { copy, results, failures, count };
	},
	$strictWebGpuRun__deps: ['$strictWebGpu', '$strictWebGpuCompile', '$strictWebGpuDevice', '$strictWebGpuWritten',
		'$strictWebGpuRecord', '$strictWebGpuLimit', '$strictWebGpuThrottled', '$strictWebGpuDestroyed', '$strictWebGpuPackedSource',
		'$strictWebGpuAwaited'],
	$strictWebGpuRun: async (name, count, inputs, output, bits) => {
		const device = inputs && (strictWebGpu.device ||= await strictWebGpuDevice());
		if (inputs && !device && !strictWebGpu.failed) {
			console.warn('WebGPU has no adapter here, GPU loops and kernels run on the CPU');
			strictWebGpu.failed = true;
		}
		if (!device)
			return 0;
		// a kernel with a variant needing 32 KB of workgroup memory (Module.strictLarger) or f16 (Module.strictHalf) takes it
		const larger = (Module['strictLarger'] || {})[name];
		if (larger && strictWebGpu.pipelines[larger] !== null && device.limits.maxComputeWorkgroupStorageSize >= 32768)
			try {
				await strictWebGpuCompile(device, larger);
				name = larger;
			} catch {
				strictWebGpu.pipelines[larger] = null;
			}
		const half = (Module['strictHalf'] || {})[name];
		if (half && strictWebGpu.pipelines[half] !== null && device.features.has('shader-f16'))
			try {
				await strictWebGpuCompile(device, half);
				name = half;
			} catch {
				strictWebGpu.pipelines[half] = null;
			}
		const subgroup = (Module['strictSubgroups'] || {})[name];
		const canSubgroup = device.features.has('subgroups') && (device.adapterInfo?.subgroupMinSize ?? 4) >= 4;
		if (subgroup && strictWebGpu.pipelines[subgroup] !== null && canSubgroup)
			try {
				await strictWebGpuCompile(device, subgroup);
				name = subgroup;
			} catch {
				strictWebGpu.pipelines[subgroup] = null;
			}
		const pipeline = await strictWebGpuCompile(device, name);
		const grouped = (Module['strictKernelGroups'] || {})[name];
		const splits = grouped ? grouped(inputs.places)[2] || 1 : 1;
		if (splits > 1)
			await strictWebGpuCompile(device, 'strict_splits');
		if (!output && bits === 17) {
			const packedMap = (Module['strictPacked'] ||= {});
			if (packedMap[name] === undefined) {
				const packedCode = !grouped && strictWebGpuPackedSource(Module['strictShaders'][name]);
				if (packedCode) {
					Module['strictShaders'][name + '_p17'] = packedCode;
					packedMap[name] = name + '_p17';
				} else
					packedMap[name] = null;
			}
			const packKey = packedMap[name];
			const canPack = packKey && splits <= 1 && strictWebGpu.pipelines[packKey] !== null &&
				(!(Module['strictCanPack'] || {})[name] || Module['strictCanPack'][name](inputs.places, count));
			if (canPack)
				try { await strictWebGpuCompile(device, packKey); } catch { strictWebGpu.pipelines[packKey] = null; }
			if (!canPack || !strictWebGpu.pipelines[packKey])
				await strictWebGpuCompile(device, 'strict_packed');
		}
		const recorded = strictWebGpuRecord(pipeline, name, count, inputs, output, bits);
		if (recorded === undefined)
			return 0;
		if (typeof recorded !== 'number') {
			await strictWebGpuAwaited(recorded.copy.mapAsync(1));
			const mapped = recorded.copy.getMappedRange();
			const isFailed = new Uint32Array(mapped, recorded.count * 4, 1)[0] !== 0;
			if (!isFailed)
				strictWebGpuWritten(new Float32Array(mapped, 0, recorded.count), output, bits);
			recorded.copy.unmap();
			[recorded.results, recorded.failures, recorded.copy].forEach(strictWebGpuDestroyed);
			return isFailed ? 0 : 1;
		}
		await strictWebGpuThrottled(0);
		return recorded;
	},
	// a compiled kernel with nothing to upload and room in the queue, recorded without unwinding wasm (strictWebGpuRun
	// does the first of each shader, and any kernel that would wait for the GPU)
	$strictWebGpuFast__deps: ['$strictWebGpu', '$strictWebGpuRecord', '$strictWebGpuKernelInputs', '$UTF8ToString',
		'$strictWebGpuLimit'],
	$strictWebGpuFast: (namePtr, count, table, slots, bits) => {
		const device = strictWebGpu.device;
		if (!device || strictWebGpu.queued > strictWebGpuLimit())
			return undefined;
		const key = UTF8ToString(namePtr);
		const entries = HEAPF64.subarray((table >>> 0) / 8, (table >>> 0) / 8 + slots * 3);
		for (let slot = 0; slot < slots; slot++)
			if (entries[slot * 3 + 2] === 255 && entries[slot * 3] in strictWebGpu.lazy)
				return undefined;
		let name = key;
		const larger = (Module['strictLarger'] || {})[key];
		if (larger && strictWebGpu.pipelines[larger] !== null && device.limits.maxComputeWorkgroupStorageSize >= 32768) {
			if (!strictWebGpu.pipelines[larger])
				return undefined;
			name = larger;
		}
		const half = (Module['strictHalf'] || {})[name];
		if (half && strictWebGpu.pipelines[half] !== null && device.features.has('shader-f16')) {
			if (!strictWebGpu.pipelines[half])
				return undefined;
			name = half;
		}
		const subgroup = (Module['strictSubgroups'] || {})[name];
		if (subgroup && strictWebGpu.pipelines[subgroup] !== null && device.features.has('subgroups') &&
			(device.adapterInfo?.subgroupMinSize ?? 4) >= 4) {
			if (!strictWebGpu.pipelines[subgroup])
				return undefined;
			name = subgroup;
		}
		if (!strictWebGpu.pipelines[name])
			return undefined;
		const inputs = strictWebGpuKernelInputs(count, table >>> 0, slots);
		if (!inputs)
			return 0;
		const grouped = (Module['strictKernelGroups'] || {})[name];
		const splits = grouped ? grouped(inputs.places)[2] || 1 : 1;
		if (splits > 1 && !strictWebGpu.pipelines['strict_splits'])
			return undefined;
		if (bits === 17) {
			const packedMap = Module['strictPacked'] || {};
			if (packedMap[name] === undefined && !grouped)
				return undefined;
			const packKey = packedMap[name];
			const canPack = packKey && splits <= 1 && strictWebGpu.pipelines[packKey] !== null &&
				(!(Module['strictCanPack'] || {})[name] || Module['strictCanPack'][name](inputs.places, count));
			if (canPack ? !strictWebGpu.pipelines[packKey] : !strictWebGpu.pipelines['strict_packed'])
				return undefined;
		}
		return strictWebGpuRecord(strictWebGpu.pipelines[name], name, count, inputs, 0, bits);
	},
	$strictWebGpuStarted__deps: ['$strictWebGpu', '$strictWebGpuRun', '$UTF8ToString', '$Asyncify', '$strictWebGpuOutOfMemory'],
	$strictWebGpuStarted: (name, count, inputs, output, bits, placed) => Asyncify.handleSleep((wakeUp) => {
		const key = UTF8ToString(name);
		const running = (placed || Promise.resolve()).then(() => strictWebGpuRun(key, count, inputs(key), output, bits));
		running.then(wakeUp, (error) => {
			if (!strictWebGpuOutOfMemory(error)) {
				console.error('WebGPU kernel ' + key + ' runs on the CPU: ' + error);
				strictWebGpu.pipelines[key] = null;
			}
			wakeUp(0);
		});
	}),
	strict_webgpu_kernel__deps: ['$strictWebGpuStarted', '$strictWebGpuKernelInputs', '$strictWebGpuPlacedInputs'],
	strict_webgpu_kernel__async: true,
	strict_webgpu_kernel: (name, count, table, slots, bits) => strictWebGpuStarted(name, count,
		() => strictWebGpuKernelInputs(count, table >>> 0, slots), 0, bits, strictWebGpuPlacedInputs(table >>> 0, slots)),
	// -1 when this kernel must wait (compile, upload or the queue is full): a synchronous return from the async import
	// left Asyncify unable to rewind the next wait. 0 is a failed kernel, a positive handle is the kept result
	strict_webgpu_kernel_fast__deps: ['$strictWebGpuFast', '$UTF8ToString', '$strictWebGpuOutOfMemory'],
	strict_webgpu_kernel_fast: (name, count, table, slots, bits) => {
		try {
			const fast = strictWebGpuFast(name, count, table, slots, bits);
			return fast === undefined ? -1 : fast;
		} catch (error) {
			if (!strictWebGpuOutOfMemory(error)) {
				console.error('WebGPU kernel ' + UTF8ToString(name) + ' runs on the CPU: ' + error);
				strictWebGpu.pipelines[UTF8ToString(name)] = null;
			}
			return 0;
		}
	},
	// the numbers of a kept kernel result (strict_gpu_web_get) as bits into target: 1, 2 when its loop read outside of a
	// list, 0 when the buffer is gone
	strict_webgpu_download__deps: ['$strictWebGpu', '$strictWebGpuWritten', '$strictWebGpuWidened', '$strictWebGpuEncoder',
		'$strictWebGpuFlush', '$Asyncify', '$strictWebGpuBuffer', '$strictWebGpuDestroyed', '$strictWebGpuPlaced',
		'$strictWebGpuAwaited', '$strictWebGpuOutOfMemory'],
	strict_webgpu_download__async: true,
	// a package shows a kept result without the download (GpuMemory.Address gives its handle, DeltaEngine Texture.Shared):
	// the copy into its own buffer is recorded behind the kernels that compute the result
	strict_webgpu_download__postset: "Module['strictGpuDevice'] = () => strictWebGpu.device; " +
		"Module['strictGpuCopied'] = (handle, target, bytes) => strictWebGpuPlaced(handle).then((source) => { " +
		"if (!source || source.size < bytes || handle in strictWebGpu.halves) return false; " +
		"strictWebGpuEncoder().copyBufferToBuffer(source, 0, target, 0, bytes); strictWebGpuFlush(); return true; });",
	strict_webgpu_download: (handle, target, count, bits) => Asyncify.handleSleep((wakeUp) =>
		strictWebGpuPlaced(handle).then((source) => {
		if (!source)
			return wakeUp(0);
		const isHalved = handle in strictWebGpu.halves, bytes = isHalved ? Math.ceil(count / 2) * 4 : count * 4;
		const copy = strictWebGpuBuffer(bytes + 4, 1 | 8), encoder = strictWebGpuEncoder();
		encoder.copyBufferToBuffer(source, 0, copy, 0, bytes);
		encoder.copyBufferToBuffer(strictWebGpu.failures[handle], 0, copy, bytes, 4);
		strictWebGpuFlush();
		strictWebGpuAwaited(copy.mapAsync(1)).then(() => {
			const mapped = copy.getMappedRange(), isOutside = new Uint32Array(mapped, bytes, 1)[0] !== 0;
			strictWebGpuWritten(isHalved ? strictWebGpuWidened(new Uint16Array(mapped, 0, count))
				: new Float32Array(mapped, 0, count), target >>> 0, bits);
			copy.unmap();
			strictWebGpuDestroyed(copy);
			wakeUp(isOutside ? 2 : 1);
		}, (error) => {
			strictWebGpuOutOfMemory(error) || console.error('WebGPU result could not be read: ' + error);
			wakeUp(0);
		});
	}, (error) => {
		strictWebGpuOutOfMemory(error) || console.error('WebGPU result could not be read: ' + error);
		wakeUp(0);
	})),
	// a new kept result of the numbers of the left one and then of the right one (strict_gpu_web_joined), 0 on failure or
	// when the halves of both would not line up (the host then joins them)
	strict_webgpu_joined__deps: ['$strictWebGpu', '$strictWebGpuEncoder', '$strictWebGpuFlag', '$strictWebGpuCreated', '$strictWebGpuOutOfMemory'],
	strict_webgpu_joined: (left, right) => {
		const first = strictWebGpu.buffers[left], second = strictWebGpu.buffers[right], device = strictWebGpu.device;
		const firstHalves = strictWebGpu.halves[left], secondHalves = strictWebGpu.halves[right];
		if (!first || !second || !device || (firstHalves === undefined) !== (secondHalves === undefined) ||
			(firstHalves !== undefined && (firstHalves === 0 || firstHalves % 2 !== 0)))
			return 0;
		let joined;
		try {
			joined = strictWebGpuCreated(first.size + second.size, 128 | 8 | 4);
		} catch (error) {
			strictWebGpuOutOfMemory(error) || console.error('Joining WebGPU buffers failed: ' + error);
			return 0;
		}
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
	strict_webgpu_uploaded__deps: ['$strictWebGpu', '$strictWebGpuDevice', '$Asyncify', '$strictWebGpuFlag',
		'$strictWebGpuThrottled', '$strictWebGpuCreated', '$strictWebGpuOutOfMemory'],
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
			const kept = strictWebGpuCreated(narrowed.byteLength, 128 | 8 | 4);
			device.queue.writeBuffer(kept, 0, narrowed);
			const handle = strictWebGpu.next++;
			strictWebGpu.buffers[handle] = kept;
			strictWebGpu.failures[handle] = strictWebGpuFlag();
			strictWebGpu.halves[handle] = words;
			strictWebGpu.uploads[handle] = words;
			strictWebGpu.grown += kept.size;
			await strictWebGpuThrottled(kept.size);
			return handle;
		}
		const isSixteen = halves === -2;
		halves = isSixteen ? words * 2 : Math.max(halves, 0);
		const uploaded = strictWebGpuCreated(Math.max(words * 4, 4), 128 | 8 | 4);
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
		await strictWebGpuThrottled(uploaded.size);
		return handle;
	})().then(wakeUp, (error) => {
		strictWebGpuOutOfMemory(error) || console.error('Uploading to WebGPU failed: ' + error);
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
			// 12 blocks (192 MB) cover a read and the 8 after it, 48 held 768 MB a phone could not spare
			if (cache.blocks.size >= 12)
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
	strict_webfile_rows__deps: ['$strictWebGpu', '$strictWebGpuDevice', '$strictWebGpuPrecompile', '$UTF8ToString',
		'$Asyncify', '$strictWebGpuFlag', '$strictWebGpuFilled'],
	strict_webfile_rows__async: true,
	strict_webfile_rows: (url, offset, rows, rowBytes, codesBytes, stride, columns, scales) => Asyncify.handleSleep((wakeUp) => {
		const path = UTF8ToString(url), copied = HEAPF32.slice(scales >>> 2, (scales >>> 2) + rows);
		(async () => {
			strictWebGpuPrecompile(strictWebGpu.device ||= await strictWebGpuDevice());
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
	$strictWebGpuFilled__deps: ['$strictWebGpu', '$strictWebFileRange', '$strictWebGpuThrottled'],
	$strictWebGpuFilled: async (buffer, at, path, offset, bytes) => {
		const queue = strictWebGpu.device.queue, isAligned = offset % 4 === 0 && at % 4 === 0;
		const whole = isAligned ? null : new Uint8Array(Math.ceil(bytes / 4) * 4);
		await strictWebFileRange(path, offset, bytes, (piece, from) => isAligned ?
			queue.writeBuffer(buffer, at + from, piece) : whole.set(piece, from));
		if (whole)
			queue.writeBuffer(buffer, at, whole);
		await strictWebGpuThrottled(bytes);
	},
	// the buffer of a kept handle: rows of a file (strict_webfile_rows), rows joined only from files and whole borrows of
	// them get it on their first use, once, again after GpuMemory.Evict gave it back (strict_webgpu_evicted)
	$strictWebGpuPlaced__deps: ['$strictWebGpu', '$strictWebFile', '$strictWebGpuCreated'],
	$strictWebGpuPlaced: (handle) => {
		const lazy = strictWebGpu.lazy[handle];
		if (!lazy)
			return Promise.resolve(strictWebGpu.buffers[handle]);
		return lazy.placing ||= (async () => {
			const buffer = lazy.borrowOf ? await strictWebGpuPlaced(lazy.borrowOf)
				: strictWebGpuCreated(lazy.size, 128 | 8 | 4);
			if (!lazy.borrowOf)
				await lazy.fill(buffer);
			if (strictWebGpu.lazy[handle] !== lazy) {
				lazy.borrowOf || buffer.destroy();
				return undefined;
			}
			delete strictWebGpu.lazy[handle];
			if (!Object.keys(strictWebGpu.lazy).length)
				strictWebFile.blocks.clear();
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
		'$strictWebGpuPlaced', '$strictWebGpuFilled', '$strictWebGpuCreated', '$strictWebGpuOutOfMemory'],
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
				const joined = strictWebGpuCreated(count * stride + count * 4, 128 | 8 | 4);
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
			strictWebGpuOutOfMemory(error) || console.error('Gathering rows on WebGPU failed: ' + error);
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
	// GpuMemory.Evict gives back the buffers of rows read from files and of rows joined only from them, their next use
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
	// GpuMemory.Trim and Evict destroy the pooled buffers of finished kernels and the free failure flags after the
	// commands still to be submitted (Firefox ran out of memory loading Moss with the encoder's buffers pooled)
	strict_webgpu_trimmed__deps: ['$strictWebGpuTrim'],
	strict_webgpu_trimmed: () => strictWebGpuTrim(),
	// not 0 once kept results took 1 GB (64 MB on mobile) since the last time. 1 asks for a full collection, 2 for a young one.
	// A young collection does not free a dead WebGPU list that already survived one pass (class 255, same as a dead old
	// Flux activation on native). On a desktop that left every past 720p step resident: three 1280x720 images climbed
	// from 6 s to 22 s while the GPU stayed above 20 GB. Weights live in filed and do not count toward that.
	strict_webgpu_crowded__deps: ['$strictWebGpu', '$strictWebGpuMobile'],
	strict_webgpu_crowded: () => {
		const isCrowded = strictWebGpu.grown > (strictWebGpuMobile() ? 67108864 : 1073741824);
		if (!isCrowded)
			return 0;
		strictWebGpu.grown = 0;
		let live = 0;
		for (const handle in strictWebGpu.buffers) {
			const buffer = strictWebGpu.buffers[handle];
			if (buffer && buffer.size && !strictWebGpu.filed[handle])
				live += buffer.size;
		}
		return strictWebGpuMobile() || live > 2147483648 ? 1 : 2;
	},
});
