using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class NativeFileRegistryTests
{
	[Test]
	public void ReadingMissingFileFailsWithoutCreatingIt()
	{
		var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".txt");
		Assert.That(() => NativeFileRegistry.ReadLines(Open(path)),
			Throws.InstanceOf<FileNotFoundException>());
		Assert.That(File.Exists(path), Is.False);
	}

	private static long Open(string path) =>
		(long)NativeFileRegistry.Open(TestPackage.Instance.GetType(Type.File), path).Number;

	[Test]
	public void ReadingDoesNotBlockOtherReaders()
	{
		var path = Path.GetTempFileName();
		File.WriteAllText(path, "has number");
		var handle = Open(path);
		try
		{
			Assert.That(NativeFileRegistry.ReadLines(handle), Is.EqualTo(new[] { "has number" }));
			Assert.That(() => File.ReadAllText(path), Throws.Nothing);
		}
		finally
		{
			NativeFileRegistry.Close(handle);
			File.Delete(path);
		}
	}
}
