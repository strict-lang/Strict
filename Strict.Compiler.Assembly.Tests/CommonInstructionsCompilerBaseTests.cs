using NUnit.Framework;

namespace Strict.Compiler.Assembly.Tests;

public sealed class CommonInstructionsCompilerBaseTests
{
	[Test]
	public void CompilersUseSharedAssemblyBaseClass()
	{
		Assert.That(typeof(InstructionsToAssembly).BaseType?.Name,
			Is.EqualTo(nameof(InstructionsToAssemblyCompiler)));
		Assert.That(typeof(InstructionsToLlvmIr).BaseType?.Name,
			Is.EqualTo(nameof(InstructionsToAssemblyCompiler)));
		Assert.That(typeof(InstructionsToMlir).BaseType?.Name,
			Is.EqualTo(nameof(InstructionsToAssemblyCompiler)));
	}
}
