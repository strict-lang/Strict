# Same blur as BenchBlur.strict and blur.c, usage: python blur.py width height radius
import sys, time
width, height, radius = (int(argument) for argument in (sys.argv[1:] + ["3840", "2160", "4"][len(sys.argv) - 1:])[:3])
length, stride = width * height * 4, width * 4
pixels = bytearray(index * 7919 % 256 for index in range(length))
blurred = bytearray(length)
area = (2 * radius + 1) * (2 * radius + 1)
started = time.perf_counter()
for y in range(height):
	for x in range(width):
		for channel in range(4):
			index = y * stride + x * 4 + channel
			if x < radius or x >= width - radius or y < radius or y >= height - radius:
				blurred[index] = pixels[index]
			else:
				total = 0
				for dy in range(-radius, radius + 1):
					for dx in range(-radius, radius + 1):
						total += pixels[index + dy * stride + dx * 4]
				blurred[index] = total // area
print(f"Python: blurred {width}x{height} with {2 * radius + 1}x{2 * radius + 1} in {(time.perf_counter() - started) * 1000:.1f} ms, pixel {blurred[length // 2 + stride // 2]}")
