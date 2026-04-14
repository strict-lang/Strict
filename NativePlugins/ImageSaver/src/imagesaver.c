/*
 * imagesaver.c — native plugin for Strict ImageSaver.strict
 *
 * Exports a single C function that NativePluginLoader calls:
 *   ImageSaver_Save(path, data, dataLength, width, height) — writes RGBA8888 pixels to file
 *
 * Format (PNG or JPG) is determined by the file extension.
 * Uses stb_image_write (single-header, public domain) for cross-platform encoding.
 *
 * Build:
 *   Linux:   gcc -shared -fPIC -O2 -o ImageSaver.so imagesaver.c -lm
 *   macOS:   gcc -dynamiclib -O2 -o ImageSaver.dylib imagesaver.c -lm
 *   Windows: gcc -shared -O2 -o ImageSaver.dll imagesaver.c
 *            (or: cl /LD /O2 imagesaver.c)
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <string.h>

/* Returns the file extension (including the dot), or NULL if none. */
static const char* GetExtension(const char* path)
{
	const char* dot = strrchr(path, '.');
	return dot;
}

/* Writes RGBA8888 pixel data to a PNG or JPG file.
 * Returns 1 on success, 0 on failure. */
#ifdef _WIN32
__declspec(dllexport)
#endif
int ImageSaver_Save(const char* path, const unsigned char* data, int width, int height)
{
	if (path == NULL || data == NULL || width <= 0 || height <= 0)
		return 0;
	int dataLength = width * height * 4;
	const char* ext = GetExtension(path);
	if (ext == NULL)
		return 0;
	if (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0)
		return stbi_write_png(path, width, height, 4, data, width * 4);
	if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 ||
		strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPEG") == 0)
		return stbi_write_jpg(path, width, height, 4, data, 90);
	return 0;
}
