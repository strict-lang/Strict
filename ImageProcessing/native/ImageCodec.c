#include "strict.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
typedef struct { UINT version; void* callback; BOOL noThread; BOOL noCodecs; } Startup;
typedef struct { GUID guid; ULONG count; ULONG type; void* value; } Parameter;
typedef struct { UINT count; Parameter parameter; } Parameters;
__declspec(dllimport) int WINAPI GdiplusStartup(ULONG_PTR*, const Startup*, void*);
__declspec(dllimport) void WINAPI GdiplusShutdown(ULONG_PTR);
__declspec(dllimport) int WINAPI GdipCreateBitmapFromScan0(int, int, int, int, BYTE*, void**);
__declspec(dllimport) int WINAPI GdipSaveImageToFile(void*, const WCHAR*, const CLSID*, const void*);
__declspec(dllimport) int WINAPI GdipDisposeImage(void*);

static double Written(const char* path, BYTE* bytes, int width, int height, int stride, ULONG quality)
{
	int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
	if (!length) return strict_fail("JpegInvalidPath");
	WCHAR* wide = malloc((size_t)length * sizeof(WCHAR));
	if (!wide) return strict_fail("JpegOutOfMemory");
	MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, length);
	Startup startup = { 1, NULL, FALSE, FALSE };
	ULONG_PTR token;
	int status = GdiplusStartup(&token, &startup, NULL);
	if (status) { free(wide); return strict_fail("JpegStartupFailed"); }
	void* bitmap = NULL;
	status = GdipCreateBitmapFromScan0(width, height, stride, 0x21808, bytes, &bitmap);
	CLSID encoder = { 0x557cf401, 0x1a04, 0x11d3, { 0x9a, 0x73, 0, 0, 0xf8, 0x1e, 0xf3, 0x2e } };
	Parameters parameters = { 1, { { 0x1d5be4b5, 0xfa4a, 0x452d,
		{ 0x9c, 0xdd, 0x5d, 0xb3, 0x51, 0x05, 0xe7, 0xeb } }, 1, 4, &quality } };
	if (!status) status = GdipSaveImageToFile(bitmap, wide, &encoder, &parameters);
	if (bitmap) GdipDisposeImage(bitmap);
	GdiplusShutdown(token);
	free(wide);
	return status ? strict_fail("JpegWriteFailed") : 1;
}
#elif defined(__EMSCRIPTEN__)
#include <emscripten.h>

/* the browser's own encoder writes the file, the page then offers it as a download (NativeRuntimeWgsl.js) */
EM_JS_DEPS(jpeg, "$FS,$UTF8ToString");
EM_ASYNC_JS(int, jpeg_encoded, (const char* path, unsigned char* bytes, int width, int height, double quality), {
	try
	{
		var canvas = new OffscreenCanvas(width, height);
		var colors = new Uint8ClampedArray(HEAPU8.subarray(bytes >>> 0, (bytes >>> 0) + width * height * 4));
		canvas.getContext("2d").putImageData(new ImageData(colors, width, height), 0, 0);
		var blob = await canvas.convertToBlob({ "type": "image/jpeg", "quality": quality / 100 });
		FS.writeFile(UTF8ToString(path), new Uint8Array(await blob.arrayBuffer()));
		return 1;
	}
	catch (error)
	{
		console.error("Saving " + UTF8ToString(path) + " failed: " + error);
		return 0;
	}
});

static double Written(const char* path, unsigned char* bytes, int width, int height, int stride, unsigned long quality)
{
	return jpeg_encoded(path, bytes, width, height, (double)quality) ? 1 : strict_fail("JpegWriteFailed");
}
#endif

STRICT_EXPORT double ImageCodec_SaveJpeg(double self, double path, double pixels, double width, double quality)
{
#if defined(_WIN32) || defined(__EMSCRIPTEN__)
	const strict_numbers* values = strict_numbers_of(pixels);
	if (!isfinite(width) || width < 1 || width > (INT_MAX - 3) / 3 || floor(width) != width ||
		!isfinite(quality) || quality < 0 || quality > 100 || floor(quality) != quality)
		return strict_fail("JpegInvalidDimensionsOrQuality");
	double height = values->count / (width * 3);
	if (!isfinite(height) || height < 1 || height > INT_MAX || floor(height) != height)
		return strict_fail("JpegInvalidPixelCount");
#ifdef _WIN32
	size_t channels = 3;
	size_t stride = ((size_t)width * 3 + 3) & ~(size_t)3;
#else
	size_t channels = 4;
	size_t stride = (size_t)width * 4;
#endif
	if (stride > SIZE_MAX / (size_t)height) return strict_fail("JpegImageTooLarge");
	unsigned char* bytes = calloc((size_t)height, stride);
	if (!bytes) return strict_fail("JpegOutOfMemory");
	for (size_t row = 0; row < (size_t)height; row++)
		for (size_t column = 0; column < (size_t)width; column++)
			for (size_t channel = 0; channel < channels; channel++)
			{
				double value = channel == 3 ? 1 : values->items[(row * (size_t)width + column) * 3 + channel];
				bytes[row * stride + column * channels + (channels == 3 ? 2 - channel : channel)] =
					!(value > 0) ? 0 : value >= 1 ? 255 : (unsigned char)(value * 255 + 0.5);
			}
	double result = Written(strict_text(path), bytes, (int)width, (int)height, (int)stride, (unsigned long)quality);
	free(bytes);
	return result;
#else
	return strict_fail("JpegRequiresWindowsOrWeb");
#endif
}
