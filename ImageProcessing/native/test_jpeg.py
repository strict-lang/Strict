"""Run with a built ImageProcessing.dll; checks codec output and errors."""
import ctypes
from pathlib import Path
import struct
import sys
import tempfile
from PIL import Image

codec = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
save = codec.ImageCodec_SaveJpeg
save.argtypes = [ctypes.c_double] * 5
save.restype = ctypes.c_double

def pointer(value):
	return struct.unpack('d', struct.pack('Q', ctypes.addressof(value)))[0]

def failed(value):
	return struct.unpack('Q', struct.pack('d', value))[0] >> 48 == 0x7FFC

with tempfile.TemporaryDirectory() as directory:
	path = Path(directory) / 'è‰².jpg'
	name = ctypes.create_string_buffer(str(path).encode('utf-8'))
	pixels = (ctypes.c_double * (1 + 17 * 12 * 3))(17 * 12 * 3, *([1, 0.25, 0] * (17 * 12)))
	assert save(0, pointer(name), pointer(pixels), 17, 85) == 1, 'JPEG save failed'
	with Image.open(path) as image:
		assert image.format == 'JPEG' and image.size == (17, 12)
		assert max(abs(left - right) for left, right in zip(image.getpixel((8, 6)), (255, 64, 0))) < 6
	assert failed(save(0, pointer(name), pointer(pixels), 0, 85))
	assert failed(save(0, pointer(name), pointer(pixels), 17, 101))
	assert failed(save(0, pointer(name), pointer(pixels), 18, 85))
	missing = ctypes.create_string_buffer(str(path / 'missing.jpg').encode('utf-8'))
	assert failed(save(0, pointer(missing), pointer(pixels), 17, 85))
print('JPEG encoding, Unicode path, channel order, padded stride and errors passed')
