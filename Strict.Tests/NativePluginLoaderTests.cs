using System.Reflection;

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

	/*
	[Test]
	[Category("Slow")]
	//TODO: wrong anyway, remove
	public void LoadImageLoaderPluginAndCallLoad()
	{
		var pluginDllPath = FindImageLoaderDll();
		if (pluginDllPath == null)
		{
			Assert.Ignore("ImageLoader plugin DLL not found — build NativePlugins/ImageLoader first");
			return;
		}
		var assembly = Assembly.LoadFrom(pluginDllPath);
		var imageType = assembly.GetExportedTypes().First(type =>
			type.Name.Equals("Image", StringComparison.OrdinalIgnoreCase));
		Assert.That(imageType, Is.Not.Null);
		var loadMethod = imageType.GetMethod("Load", BindingFlags.Public | BindingFlags.Static);
		Assert.That(loadMethod, Is.Not.Null);
		Assert.That(loadMethod!.GetParameters().Length, Is.EqualTo(1));
		Assert.That(loadMethod.ReturnType, Is.EqualTo(typeof(byte[])));
	}

	private static string? FindImageLoaderDll()
	{
		var repoRoot = FindRepoRoot();
		if (repoRoot == null)
			return null;
		var dllPath = Path.Combine(repoRoot, "NativePlugins", "ImageLoader", "bin", "Debug",
			"net10.0", "ImageLoader.dll");
		return File.Exists(dllPath)
			? dllPath
			: null;
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
	*/
}
