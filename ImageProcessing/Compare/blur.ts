// Same blur as BenchBlur.strict and blur.c, usage: node blur.ts width height radius
const [width, height, radius] = [3840, 2160, 4].map((fallback, index) => Number(process.argv[2 + index] ?? fallback));
const length = width * height * 4, stride = width * 4;
const pixels = new Uint8Array(length);
for (let index = 0; index < length; index++) pixels[index] = index * 7919 % 256;
const blurred = new Uint8Array(length);
const area = (2 * radius + 1) * (2 * radius + 1);
const started = performance.now();
for (let y = 0; y < height; y++)
	for (let x = 0; x < width; x++)
		for (let channel = 0; channel < 4; channel++) {
			const index = y * stride + x * 4 + channel;
			if (x < radius || x >= width - radius || y < radius || y >= height - radius)
				blurred[index] = pixels[index];
			else {
				let sum = 0;
				for (let dy = -radius; dy <= radius; dy++)
					for (let dx = -radius; dx <= radius; dx++)
						sum += pixels[index + dy * stride + dx * 4];
				blurred[index] = Math.floor(sum / area);
			}
		}
console.log(`TypeScript: blurred ${width}x${height} with ${2 * radius + 1}x${2 * radius + 1} in ${(performance.now() - started).toFixed(1)} ms, pixel ${blurred[length / 2 + stride / 2]}`);
