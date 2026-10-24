#include "strict.h"
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
#endif

STRICT_EXPORT double ImageCodec_SaveJpeg(double self, double path, double pixels, double width, double quality)
{
#ifdef _WIN32
	const strict_numbers* values = strict_numbers_of(pixels);
	if (!isfinite(width) || width < 1 || width > (INT_MAX - 3) / 3 || floor(width) != width ||
		!isfinite(quality) || quality < 0 || quality > 100 || floor(quality) != quality)
		return strict_fail("JpegInvalidDimensionsOrQuality");
	double height = values->count / (width * 3);
	if (!isfinite(height) || height < 1 || height > INT_MAX || floor(height) != height)
		return strict_fail("JpegInvalidPixelCount");
	size_t stride = ((size_t)width * 3 + 3) & ~(size_t)3;
	if (stride > SIZE_MAX / (size_t)height) return strict_fail("JpegImageTooLarge");
	BYTE* bytes = calloc((size_t)height, stride);
	if (!bytes) return strict_fail("JpegOutOfMemory");
	for (size_t row = 0; row < (size_t)height; row++)
		for (size_t column = 0; column < (size_t)width; column++)
			for (size_t channel = 0; channel < 3; channel++)
			{
				double value = values->items[(row * (size_t)width + column) * 3 + channel];
				bytes[row * stride + column * 3 + 2 - channel] =
					!(value > 0) ? 0 : value >= 1 ? 255 : (BYTE)(value * 255 + 0.5);
			}
	double result = Written(strict_text(path), bytes, (int)width, (int)height, (int)stride, (ULONG)quality);
	free(bytes);
	return result;
#else
	return strict_fail("JpegRequiresWindows");
#endif
}
