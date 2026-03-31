# Native Image Loader Plugin

Example .NET plugin for the Strict `NativePluginLoader`. Implements the `Image.strict` trait method `Load(path Text) Bytes` using SkiaSharp for cross-platform JPG/PNG/BMP/GIF/WebP decoding.

## How It Works

`Image.strict` declares a trait method (no body):
```
Load(path Text) Bytes
```

When the VM encounters this trait call, `NativePluginLoader` searches DLLs in the current directory for a class named `Image` with a method named `Load` taking one argument. This plugin provides exactly that — an `Image.Load(string path)` method returning raw RGB `byte[]`.

## Build

```bash
cd NativePlugins/ImageLoader
dotnet build
```

## Deploy

Copy the output DLL and its SkiaSharp dependencies to the directory where you run `.strict` files:

```bash
cp bin/Debug/net10.0/ImageLoader.dll /path/to/working/dir/
cp bin/Debug/net10.0/SkiaSharp.dll /path/to/working/dir/
cp bin/Debug/net10.0/libSkiaSharp.* /path/to/working/dir/  # native library
```

Or publish as self-contained:
```bash
dotnet publish -c Release
cp bin/Release/net10.0/publish/*.dll /path/to/working/dir/
```

## Output Format

`Image.Load` returns raw RGB bytes — 3 bytes per pixel (Red, Green, Blue) in row-major order. This is compatible with `ColorImage.BytesToColors` which divides each byte by 255 to get normalized 0–1 color values.

## Writing Your Own Plugin

Any .NET class library can serve as a plugin. The rules are:
1. Class name must match the Strict type name (case-insensitive)
2. Method name must match the Strict method name (case-insensitive)
3. Parameter count must match
4. Arguments arrive as `string` (for Text), `double` (for Number), or `object?[]` (for Lists)
5. Return `byte[]`, `string`, `double`, `bool`, or `null`
