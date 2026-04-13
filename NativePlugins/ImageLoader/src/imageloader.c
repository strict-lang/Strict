/*
 * imageloader.c — native plugin for Strict ImageLoader.strict
 *
 * Exports three C functions that NativePluginLoader calls:
 *   ImageLoader_Create(path)   — loads the image, returns an opaque handle
 *   ImageLoader_Colors(handle) — returns RGBA8888 byte data + count
 *   ImageLoader_Delete(handle) — frees the native memory
 *
 * Uses stb_image (single-header, public domain) for PNG/JPG/BMP/GIF/TGA support.
 *
 * Build:
 *   Linux:   gcc -shared -fPIC -O2 -o ImageLoader.so imageloader.c -lm
 *   macOS:   gcc -dynamiclib -O2 -o ImageLoader.dylib imageloader.c -lm
 *   Windows: gcc -shared -O2 -o ImageLoader.dll imageloader.c
 *            (or: cl /LD /O2 imageloader.c)
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdlib.h>
#include <string.h>

/* Opaque handle that stores the decoded image pixels. */
typedef struct ImageHandle
{
	unsigned char* data;
	int width;
	int height;
	int channels;
	int byteCount;
} ImageHandle;

/* Creates a loader for the image at the given path.
 * Returns a non-null handle on success, NULL if the file cannot be read or decoded. */
void* ImageLoader_Create(const char* path)
{
	int width, height, channels;
	/* Request 4 channels (RGBA8888) regardless of source format. */
	unsigned char* pixels = stbi_load(path, &width, &height, &channels, 4);
	if (pixels == NULL)
		return NULL;
	ImageHandle* handle = (ImageHandle*)malloc(sizeof(ImageHandle));
	if (handle == NULL)
	{
		stbi_image_free(pixels);
		return NULL;
	}
	handle->data = pixels;
	handle->width = width;
	handle->height = height;
	handle->channels = 4;
	handle->byteCount = width * height * 4;
	return handle;
}

/* Returns a pointer to the RGBA8888 pixel bytes and sets *outCount to the byte count.
 * The caller must NOT free this pointer — call ImageLoader_Delete to release it.
 * Returns NULL if the handle is invalid. */
const unsigned char* ImageLoader_Colors(void* opaqueHandle, int* outCount)
{
	if (opaqueHandle == NULL || outCount == NULL)
		return NULL;
	ImageHandle* handle = (ImageHandle*)opaqueHandle;
	*outCount = handle->byteCount;
	return handle->data;
}

/* Frees the native image memory.  Must be called exactly once after ImageLoader_Colors. */
void ImageLoader_Delete(void* opaqueHandle)
{
	if (opaqueHandle == NULL)
		return;
	ImageHandle* handle = (ImageHandle*)opaqueHandle;
	stbi_image_free(handle->data);
	free(handle);
}
