using SkiaSharp;

namespace ImageLoader;

/// <summary>
/// Native plugin implementation for Image.strict's Load(path Text) Bytes trait method.
/// The NativePluginLoader matches class name "Image" and method name "Load" by reflection.
/// Returns raw RGB byte triplets (3 bytes per pixel: R, G, B) suitable for
/// ColorImage.BytesToColors conversion in Strict.
/// Build: dotnet build, then copy ImageLoader.dll + SkiaSharp*.dll to the working directory
/// where you run your .strict files from.
/// </summary>
public class Image
{
	/// <summary>
	/// Loads an image file (.jpg, .png, .bmp, .gif, .webp) and returns raw RGB bytes.
	/// Each pixel produces 3 bytes (Red, Green, Blue) in row-major order.
	/// The returned byte[] length equals width * height * 3.
	/// </summary>
	public static byte[] Load(string path)
	{
		if (!File.Exists(path))
			throw new FileNotFoundException("Image file not found: " + path);
		using var stream = File.OpenRead(path);
		using var codec = SKCodec.Create(stream) ??
			throw new InvalidOperationException("Cannot decode image: " + path);
		var info = new SKImageInfo(codec.Info.Width, codec.Info.Height, SKColorType.Rgba8888,
			SKAlphaType.Unpremul);
		using var bitmap = new SKBitmap(info);
		codec.GetPixels(info, bitmap.GetPixels());
		var pixelCount = info.Width * info.Height;
		var rgbBytes = new byte[pixelCount * 3];
		var pixels = bitmap.GetPixelSpan();
		for (var pixelIndex = 0; pixelIndex < pixelCount; pixelIndex++)
		{
			var sourceOffset = pixelIndex * 4;
			var destOffset = pixelIndex * 3;
			rgbBytes[destOffset] = pixels[sourceOffset];
			rgbBytes[destOffset + 1] = pixels[sourceOffset + 1];
			rgbBytes[destOffset + 2] = pixels[sourceOffset + 2];
		}
		return rgbBytes;
	}
}
