# Native Image Loader Plugin

A **true native C plugin** for the Strict `NativePluginLoader`. Implements `ImageLoader.strict`'s
`from(path)` / `Colors Bytes` trait using [stb_image](https://github.com/nothings/stb) for
cross-platform JPG / PNG / BMP / GIF / TGA decoding. No .NET or JVM dependency.

## How It Works

`ImageLoader.strict` declares a trait:
```
from(path)
Colors Bytes
```

`NativePluginLoader` looks for a platform-native shared library (`ImageLoader.dll` /
`ImageLoader.so` / `ImageLoader.dylib`) in the working directory and resolves three C-ABI
functions by name:

| C function | Called when |
|---|---|
| `ImageLoader_Create(path)` | `ImageLoader.from(path)` in Strict |
| `ImageLoader_Colors(handle, &count)` | `.Colors` on an ImageLoader instance |
| `ImageLoader_Delete(handle)` | automatically after Colors to free native memory |

The pixel data is returned as RGBA8888 bytes — 4 bytes per pixel (R, G, B, A) in row-major
order. The Strict runtime copies the bytes and then immediately calls `Delete` so you never
need to manage the native memory from Strict code.

## Build

### One-command build (Linux / macOS)

```bash
cd NativePlugins/ImageLoader
sh build.sh          # produces ImageLoader.so or ImageLoader.dylib
```

### Windows (cmd)

```bat
cd NativePlugins\ImageLoader
build.bat            # uses cl.exe (MSVC) or gcc (MinGW)
```

### Cross-platform via CMake

```bash
cmake -B build
cmake --build build
# output: build/ImageLoader.{so,dll,dylib}
```

## Deploy

Copy the built shared library to the directory where you run `.strict` files from.
`NativePluginLoader` searches the current working directory.

```bash
cp ImageLoader.so /path/to/working/dir/   # Linux
# or
cp ImageLoader.dylib /path/to/working/dir/  # macOS
# or
copy ImageLoader.dll C:\path\to\working\dir\  # Windows
```

## Writing Your Own Native Plugin

The convention is straightforward:

```c
// Create is called for from(path) — return NULL on error
void* MyType_Create(const char* path);

// Colors is called to retrieve bytes — set *outCount to byte count
const uint8_t* MyType_Colors(void* handle, int* outCount);

// Delete is called automatically after Colors to free native memory
void MyType_Delete(void* handle);
```

Compile to a shared library named `MyType.dll` / `MyType.so` / `MyType.dylib` and place it
next to your `.strict` files.  Strict will find and call it automatically when encountering
`MyType.from(path)` and `.Colors` in the source.
