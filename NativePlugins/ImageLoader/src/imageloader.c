/*
 * imageloader.c — native plugin for Strict ImageLoader.strict
 *
 * Exports three C functions that NativePluginLoader calls:
 *   ImageLoader_Create(path)   — loads the image, returns an opaque handle
 *   ImageLoader_Colors(handle) — returns RGBA8888 byte data + width + height
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
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

#include <stdlib.h>
#include <string.h>

typedef struct ImageHandle
{
	unsigned char* data;
	int width;
	int height;
} ImageHandle;

/* Creates a loader for the image at the given path.
 * Returns a non-null handle on success, NULL if the file cannot be read or decoded. */
#ifdef _WIN32
__declspec(dllexport)
#endif
void* ImageLoader_Create(const char* path)
{
	int width, height, channels;
	/* Request 4 channels (RGBA8888) regardless of source format. */
	unsigned char* pixels = stbi_load(path, &width, &height, &channels, 4);
	if (pixels == NULL)
		return NULL;
	ImageHandle* handle = (ImageHandle*)malloc(sizeof(ImageHandle));
	handle->data = pixels;
	handle->width = width;
	handle->height = height;
	return handle;
}

/* Returns a pointer to the RGBA8888 pixel bytes and width, height.
 * The caller must NOT free this pointer — call ImageLoader_Delete to release it. */
#ifdef _WIN32
__declspec(dllexport)
#endif
const unsigned char* ImageLoader_Colors(void* imageHandle, int* outWidth, int* outHeight)
{
	if (imageHandle == NULL || outWidth == NULL || outHeight == NULL)
		return NULL;
	ImageHandle* handle = (ImageHandle*)imageHandle;
	*outWidth = handle->width;
	*outHeight = handle->height;
	return handle->data;
}

/* Frees the native image memory. Must be called exactly once after ImageLoader_Colors. */
#ifdef _WIN32
__declspec(dllexport)
#endif
void ImageLoader_Delete(void* imageHandle)
{
	if (imageHandle == NULL)
		return;
	ImageHandle* handle = (ImageHandle*)imageHandle;
	stbi_image_free(handle->data);
	free(handle);
}