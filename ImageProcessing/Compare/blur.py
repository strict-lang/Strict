# Same blur as BenchBlur.strict on Landscape.ppm, python blur.py (about 5 minutes)
import time
data = open("Landscape.ppm", "rb").read()
fields = data.split(maxsplit=4)
width, height = int(fields[1]), int(fields[2])
length, stride, radius = width * height * 3, width * 3, 4
pixels = data[len(data) - length:]
blurred = bytearray(length)
area = (2 * radius + 1) * (2 * radius + 1)
started = time.perf_counter()
for y in range(height):
	for x in range(width):
		for channel in range(3):
			index = y * stride + x * 3 + channel
			if x < radius or x >= width - radius or y < radius or y >= height - radius:
				blurred[index] = pixels[index]
			else:
				total = 0
				for dy in range(-radius, radius + 1):
					for dx in range(-radius, radius + 1):
						total += pixels[index + dy * stride + dx * 3]
				blurred[index] = total // area
print(f"Python: blurred {width}x{height} in {(time.perf_counter() - started) * 1000:.1f} ms")
open("Landscape.python.ppm", "wb").write(b"P6\n%d %d\n255\n" % (width, height) + blurred)
