using System.Runtime.InteropServices;

namespace Strict.Tests;

public sealed class NativePluginLoaderTests
{
	[Test]
	public void ThrowsWhenNoDllFound() =>
		Assert.That(
			() => NativePluginLoader.TryCallNativeMethod("Image", "Load", ["test.png"],
				"/nonexistent"),
			Throws.TypeOf<NativePluginLoader.NativeMethodNotFound>());

	[Test]
	public void ThrowsWhenSearchDirectoryDoesNotExist() =>
		Assert.That(
			() => NativePluginLoader.TryCallNativeMethod("NoType", "NoMethod", [],
				Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString())),
			Throws.TypeOf<NativePluginLoader.NativeMethodNotFound>());

	[Test]
	public void TryLoadNativeLifecycleReturnsNullWhenNoLibraryFound() =>
		Assert.That(
			NativePluginLoader.TryLoadNativeLifecycle("ImageLoader", "any.png",
				Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString())),
			Is.Null);

	[Test]
	[Category("Slow")]
	public void LoadNativeImageLoaderPluginAndGetRgbaBytes()
	{
		var libPath = FindBuiltNativeLibrary();
		if (libPath == null)
		{
			Assert.Ignore(
				"Native ImageLoader library not found — build NativePlugins/ImageLoader first:\n" +
				"  Linux:  gcc -shared -fPIC -O2 -o ImageLoader.so src/imageloader.c -Isrc -lm\n" +
				"  macOS:  gcc -dynamiclib -O2 -o ImageLoader.dylib src/imageloader.c -Isrc\n" +
				"  Win:    gcc -shared -O2 -o ImageLoader.dll src/imageloader.c -Isrc");
			return;
		}
		var testImage = CreateMinimalPng();
		var imageFile = Path.GetTempFileName() + ".png";
		File.WriteAllBytes(imageFile, testImage);
		try
		{
			var tempDir = Path.GetDirectoryName(imageFile)!;
			// Copy the library next to the temp image so the loader finds it
			var libDest = Path.Combine(tempDir, Path.GetFileName(libPath));
			File.Copy(libPath, libDest, overwrite: true);
			var bytes = NativePluginLoader.TryLoadNativeLifecycle("ImageLoader", imageFile, tempDir);
			Assert.That(bytes, Is.Not.Null);
			Assert.That(bytes!.Length, Is.EqualTo(4), "1×1 RGBA image = 4 bytes");
		}
		finally
		{
			File.Delete(imageFile);
		}
	}

	private static string? FindBuiltNativeLibrary()
	{
		var repoRoot = FindRepoRoot();
		if (repoRoot == null)
			return null;
		var pluginDir = Path.Combine(repoRoot, "NativePlugins", "ImageLoader");
		var candidates = new[]
		{
			Path.Combine(pluginDir, "ImageLoader.so"),
			Path.Combine(pluginDir, "ImageLoader.dylib"),
			Path.Combine(pluginDir, "ImageLoader.dll")
		};
		return candidates.FirstOrDefault(File.Exists);
	}

	private static string? FindRepoRoot()
	{
		var directory = AppContext.BaseDirectory;
		while (directory != null)
		{
			if (File.Exists(Path.Combine(directory, "Strict.sln")))
				return directory;
			directory = Path.GetDirectoryName(directory);
		}
		return null;
	}

	// Minimal valid 1×1 white PNG with an alpha channel (RGBA)
	private static byte[] CreateMinimalPng() =>
		Convert.FromBase64String(
			"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg==");
}
