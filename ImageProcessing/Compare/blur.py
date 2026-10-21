# Same 9x9 blur as BenchBlur.strict on Landscape.ppm, python blur.py (a few minutes)
import time
file = open("Landscape.ppm", "rb").read()
fields = file.split(maxsplit=4)
width, height = int(fields[1]), int(fields[2])
rgb = file[len(file) - width * height * 3:]
data = bytearray(width * height * 4)
for pixel in range(width * height):
	data[pixel * 4:pixel * 4 + 4] = rgb[pixel * 3:pixel * 3 + 3] + b"\xff"
blurred = bytearray(width * height * 4)
started = time.perf_counter()
for y in range(height):
	for x in range(width):
		red = green = blue = alpha = count = 0
		for dy in range(-4, 5):
			for dx in range(-4, 5):
				if 0 <= x + dx < width and 0 <= y + dy < height:
					source_index = ((y + dy) * width + x + dx) * 4
					red += data[source_index]
					green += data[source_index + 1]
					blue += data[source_index + 2]
					alpha += data[source_index + 3]
					count += 1
		target_index = (y * width + x) * 4
		blurred[target_index] = red // count
		blurred[target_index + 1] = green // count
		blurred[target_index + 2] = blue // count
		blurred[target_index + 3] = alpha // count
print(f"Python: blurred {width}x{height} in {(time.perf_counter() - started) * 1000:.1f} ms")
open("Landscape.python.ppm", "wb").write(b"P6\n%d %d\n255\n" % (width, height) + bytes(value for index, value in enumerate(blurred) if index % 4 != 3))
