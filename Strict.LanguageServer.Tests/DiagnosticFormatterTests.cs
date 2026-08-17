using NUnit.Framework;

namespace Strict.LanguageServer.Tests;

public sealed class DiagnosticFormatterTests
{
	[Test]
	public void ExtractDetailKeepsInstructionExecutionFailedReason()
	{
		const string message =
			"FieldLoad on non-struct value for field 'value'\n   in Strict/Boolean.not\n   Instructions (0/3):\n   >>>    0: FieldLoad value  (:line 4)\n   at Strict/Boolean.not in C:\\repo\\Boolean.strict:line 4";
		Assert.That(DiagnosticFormatter.ExtractDetail(message),
			Is.EqualTo("FieldLoad on non-struct value for field 'value'\nin Strict/Boolean.not"));
	}

	[Test]
	public void FormatMessageIncludesHumanizedCodeAndReason() =>
		Assert.That(
			DiagnosticFormatter.FormatMessage("InstructionExecutionFailed",
				"FieldLoad on non-struct value for field 'value'\n   at X in C:\\x.strict:line 4"),
			Is.EqualTo("Instruction execution failed: FieldLoad on non-struct value for field 'value'"));

	[Test]
	public void FromExceptionIncludesInnerExceptionAndTypeName()
	{
		var exception = new InvalidOperationException("outer reason",
			new ArgumentException("inner reason"));
		var diagnostic = DiagnosticFormatter.FromException(exception, ["has number"]);
		Assert.That(diagnostic.Message, Does.Contain("outer reason"));
		Assert.That(diagnostic.Message, Does.Contain("inner reason"));
		Assert.That(diagnostic.Message, Does.Contain("System.InvalidOperationException"));
	}

	[Test]
	public void ExtractDetailDropsStackOnlyMessages() =>
		Assert.That(
			DiagnosticFormatter.ExtractDetail(
				"\n   at Strict/Boolean in C:\\repo\\Boolean.strict:line 30\n"), Is.EqualTo(""));

	[Test]
	public void WindowsDocumentUriPathBecomesFileSystemPath() =>
		Assert.That("/c:/code/Strict/Boolean.strict".ToFileSystemPath(),
			Is.EqualTo("c:" + Path.DirectorySeparatorChar + "code" + Path.DirectorySeparatorChar +
				"Strict" + Path.DirectorySeparatorChar + "Boolean.strict"));

	[Test]
	public void StrictBinaryCacheIsFreshWhenBinaryIsAtLeastAsNewAsSource()
	{
		Assert.That(StrictBinaryCache.IsFresh(100, null), Is.False);
		Assert.That(StrictBinaryCache.IsFresh(100, 99), Is.False);
		Assert.That(StrictBinaryCache.IsFresh(100, 100), Is.True);
		Assert.That(StrictBinaryCache.IsFresh(100, 150), Is.True);
	}
}