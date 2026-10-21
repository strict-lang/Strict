# Same 9x9 blur as blur.py with NumPy, every offset of the window added for all pixels at once, python blur_numpy.py
import time
import numpy
file = open("Landscape.ppm", "rb").read()
fields = file[:40].split()
width, height = int(fields[1]), int(fields[2])
rgb = numpy.frombuffer(file[len(file) - width * height * 3:], numpy.uint8).reshape(height, width, 3)
data = numpy.concatenate((rgb, numpy.full((height, width, 1), 255, numpy.uint8)), axis=2).astype(numpy.int32)
started = time.perf_counter()
sums = numpy.zeros((height, width, 4), numpy.int32)
counts = numpy.zeros((height, width, 1), numpy.int32)
for dy in range(-4, 5):
	for dx in range(-4, 5):
		top, bottom = max(0, -dy), min(height, height - dy)
		left, right = max(0, -dx), min(width, width - dx)
		sums[top:bottom, left:right] += data[top + dy:bottom + dy, left + dx:right + dx]
		counts[top:bottom, left:right] += 1
blurred = (sums // counts).astype(numpy.uint8)
print(f"Python NumPy: blurred {width}x{height} in {(time.perf_counter() - started) * 1000:.1f} ms")
open("Landscape.numpy.ppm", "wb").write(f"P6\n{width} {height}\n255\n".encode() + blurred[:, :, :3].tobytes())
