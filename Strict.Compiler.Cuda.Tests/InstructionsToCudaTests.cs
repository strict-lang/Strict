using NUnit.Framework;
using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Cuda.Tests;

public sealed class InstructionsToCudaTests
{
	private readonly InstructionsToCuda compiler = new();

	[TestCase("Add", "+")]
	[TestCase("Subtract", "-")]
	[TestCase("Multiply", "*")]
	public void GenerateArrayOperationKernel(string methodName, string op)
	{
		var method = CreateSingleMethod(methodName,
			"has dummy Number",
			methodName + "(first Number, second Number) Number",
			"\tfirst " + op + " second");
		var cuda = compiler.Compile(method);
		Assert.That(cuda, Does.Contain($"void {methodName}(const float *first, const float *second, float *output, const int count)"));
		Assert.That(cuda, Does.Contain($"output[idx] = first[idx] {op} second[idx]"));
	}

	[Test]
	public void GenerateScalarOutputKernel()
	{
		var method = CreateSingleMethod("InitializeDepths",
			"has dummy Number",
			"Process(input Number, width Number, height Number, initialDepth Number) Number",
			"\tinitialDepth");
		var cuda = compiler.Compile(method);
		Assert.That(cuda, Does.Contain("void Process(const float *input, const int width, const int height, const float initialDepth, float *output)"));
		Assert.That(cuda, Does.Contain("output[idx] = initialDepth"));
	}

	[Test]
	public void KernelContainsCudaThreadIndexing()
	{
		var method = CreateSingleMethod("VectorAdd",
			"has dummy Number",
			"Add(first Number, second Number) Number",
			"\tfirst + second");
		var cuda = compiler.Compile(method);
		Assert.That(cuda, Does.Contain("blockIdx.x * blockDim.x + threadIdx.x"));
		Assert.That(cuda, Does.Contain("blockIdx.y * blockDim.y + threadIdx.y"));
		Assert.That(cuda, Does.Contain("int idx = y * blockDim.x + x"));
	}

	private static Method CreateSingleMethod(string typeName, params string[] methodLines) =>
		new Type(TestPackage.Instance, new TypeLines(typeName, methodLines)).
			ParseMembersAndMethods(new MethodExpressionParser()).Methods[0];
}
