// Same 9x9 blur as BenchBlur.strict on Landscape.ppm, node blur.ts
import { readFileSync, writeFileSync } from "node:fs";
const file = readFileSync("Landscape.ppm");
const fields = file.subarray(0, 40).toString("latin1").split(/\s+/);
const width = Number(fields[1]), height = Number(fields[2]);
const rgb = file.subarray(file.length - width * height * 3);
const data = new Uint8Array(width * height * 4);
for (let pixel = 0; pixel < width * height; ++pixel)
	data.set([rgb[pixel * 3], rgb[pixel * 3 + 1], rgb[pixel * 3 + 2], 255], pixel * 4);
const blurred = new Uint8Array(width * height * 4);
const started = performance.now();
for (let y = 0; y < height; ++y)
	for (let x = 0; x < width; ++x) {
		let red = 0, green = 0, blue = 0, alpha = 0, count = 0;
		for (let dy = -4; dy <= 4; ++dy)
			for (let dx = -4; dx <= 4; ++dx)
				if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) {
					const sourceIndex = ((y + dy) * width + x + dx) * 4;
					red += data[sourceIndex];
					green += data[sourceIndex + 1];
					blue += data[sourceIndex + 2];
					alpha += data[sourceIndex + 3];
					++count;
				}
		const targetIndex = (y * width + x) * 4;
		blurred[targetIndex] = Math.floor(red / count);
		blurred[targetIndex + 1] = Math.floor(green / count);
		blurred[targetIndex + 2] = Math.floor(blue / count);
		blurred[targetIndex + 3] = Math.floor(alpha / count);
	}
console.log(`TypeScript: blurred ${width}x${height} in ${(performance.now() - started).toFixed(1)} ms`);
writeFileSync("Landscape.typescript.ppm", Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), blurred.filter((_, index) => index % 4 !== 3)]));
