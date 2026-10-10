using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class NativeFileRegistryTests
{
	[Test]
	public void ReadingMissingFileFailsWithoutCreatingIt()
	{
		var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".txt");
		var file = NativeFileRegistry.Open(TestPackage.Instance.GetType(Type.File), path);
		Assert.That(() => NativeFileRegistry.ReadLines((long)file.Number),
			Throws.InstanceOf<FileNotFoundException>());
		Assert.That(File.Exists(path), Is.False);
	}
}
