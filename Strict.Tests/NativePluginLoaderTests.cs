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
}
