# Native Image Saver Plugin

A **native C plugin** for the Strict `NativePluginLoader`. Implements `ImageSaver.strict`'s
`Save(path, bytes, width, height)` trait using [stb_image_write](https://github.com/nothings/stb)
for cross-platform PNG / JPG encoding. No dependencies.

## How It Works

`ImageSaver.strict` declares a trait:
```
Save(path, colors Bytes, width Number, height Number)
```

`NativePluginLoader` looks for a platform-native shared library (`ImageSaver.dll` /
`ImageSaver.so` / `ImageSaver.dylib`) in the working directory and resolves the C-ABI function:

| C function | Called when |
|---|---|
| `ImageSaver_Save(path, data, len, w, h)` | `ImageSaver.Save(...)` in Strict |

The pixel data is expected as RGBA8888 bytes — 4 bytes per pixel (R, G, B, A) in row-major
order. The output format (PNG or JPG) is determined by the file extension.

## Build

### One-command build (Linux / macOS)

```bash
cd NativePlugins/ImageSaver
sh build.sh          # produces ImageSaver.so or ImageSaver.dylib
```

### Windows (cmd)

```bat
cd NativePlugins\ImageSaver
build.bat            # uses cl.exe (MSVC) or gcc (MinGW)
```

### Cross-platform via CMake

```bash
cmake -B build
cmake --build build
# output: build/ImageSaver.{so,dll,dylib}
```

## Deploy

Copy the built shared library to the directory where you run `.strict` files from.
`NativePluginLoader` searches the current working directory.
