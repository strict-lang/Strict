// Same blur as BenchBlur.strict on Landscape.ppm, node blur.ts
import { readFileSync, writeFileSync } from "node:fs";
const data = readFileSync("Landscape.ppm");
const fields = data.subarray(0, 40).toString("latin1").split(/\s+/);
const width = Number(fields[1]), height = Number(fields[2]);
const length = width * height * 3, stride = width * 3, radius = 4;
const pixels = data.subarray(data.length - length);
const blurred = new Uint8Array(length);
const area = (2 * radius + 1) * (2 * radius + 1);
const started = performance.now();
for (let y = 0; y < height; y++)
	for (let x = 0; x < width; x++)
		for (let channel = 0; channel < 3; channel++) {
			const index = y * stride + x * 3 + channel;
			if (x < radius || x >= width - radius || y < radius || y >= height - radius)
				blurred[index] = pixels[index];
			else {
				let sum = 0;
				for (let dy = -radius; dy <= radius; dy++)
					for (let dx = -radius; dx <= radius; dx++)
						sum += pixels[index + dy * stride + dx * 3];
				blurred[index] = Math.floor(sum / area);
			}
		}
console.log(`TypeScript: blurred ${width}x${height} in ${(performance.now() - started).toFixed(1)} ms`);
writeFileSync("Landscape.typescript.ppm", Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), blurred]));
