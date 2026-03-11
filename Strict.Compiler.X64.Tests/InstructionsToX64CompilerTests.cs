using NUnit.Framework;
using Strict.Compiler.X64;
using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Compiler.X64.Tests;

public sealed class InstructionsToX64CompilerTests
{
	private readonly InstructionsToX64Compiler compiler = new();

	[TestCase("Add", "+", "addsd")]
	[TestCase("Subtract", "-", "subsd")]
	[TestCase("Multiply", "*", "mulsd")]
	public void GenerateArithmeticFunction(string methodName, string op, string asmOp)
	{
		var method = CreateSingleMethod(methodName,
			"has dummy Number",
			$"{methodName}(first Number, second Number) Number",
			$"\tfirst {op} second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain($"global {methodName}"));
		Assert.That(assembly, Does.Contain($"{methodName}:"));
		Assert.That(assembly, Does.Contain(asmOp));
		Assert.That(assembly, Does.Contain("ret"));
	}

	[Test]
	public void GenerateDivideFunction()
	{
		var method = CreateSingleMethod("DivideType",
			"has dummy Number",
			"Divide(numerator Number, denominator Number) Number",
			"\tnumerator / denominator");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("divsd"));
		Assert.That(assembly, Does.Contain("ret"));
	}

	[Test]
	public void FunctionHasTextSectionAndPrologueAndEpilogue()
	{
		var method = CreateSingleMethod("SumType",
			"has dummy Number",
			"Sum(first Number, second Number) Number",
			"\tfirst + second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("section .text"));
		Assert.That(assembly, Does.Contain("push rbp"));
		Assert.That(assembly, Does.Contain("mov rbp, rsp"));
		Assert.That(assembly, Does.Contain("sub rsp,"));
		Assert.That(assembly, Does.Contain("add rsp,"));
		Assert.That(assembly, Does.Contain("pop rbp"));
		Assert.That(assembly, Does.Contain("ret"));
	}

	[Test]
	public void ParametersAreStoredToStackInPrologue()
	{
		var method = CreateSingleMethod("TotalType",
			"has dummy Number",
			"Total(first Number, second Number) Number",
			"\tfirst + second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("movsd [rbp-8], xmm0"));
		Assert.That(assembly, Does.Contain("movsd [rbp-16], xmm1"));
	}

	[Test]
	public void ReturnValueMovedToXmm0WhenNotAlreadyThere()
	{
		var method = CreateSingleMethod("SumNumbers",
			"has dummy Number",
			"SumTwo(first Number, second Number) Number",
			"\tfirst + second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("movsd xmm0,"));
	}

	[Test]
	public void NonZeroConstantGoesToDataSection()
	{
		var method = CreateSingleMethod("AddFiveType",
			"has dummy Number",
			"AddFive(first Number, second Number) Number",
			"\tfirst + 5");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("section .data"));
		Assert.That(assembly, Does.Contain("dq 0x"));
		Assert.That(assembly, Does.Contain("[rel const_"));
	}

	[Test]
	public void ZeroConstantUsesXorpd()
	{
		var method = CreateSingleMethod("ZeroAddType",
			"has dummy Number",
			"ZeroAdd(first Number, second Number) Number",
			"\tfirst + 0");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("xorpd"));
	}

	[Test]
	public void LocalVariableIsStoredAndLoadedFromStack()
	{
		var method = CreateSingleMethod("CalcType",
			"has dummy Number",
			"Calculate(first Number, second Number) Number",
			"\tlet result = first + second",
			"\tresult + 1");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("movsd [rbp-"));
		Assert.That(assembly, Does.Contain("movsd xmm"));
		Assert.That(assembly, Does.Contain("addsd"));
	}

	private static Method CreateSingleMethod(string typeName, params string[] methodLines) =>
		new Type(TestPackage.Instance, new TypeLines(typeName, methodLines)).
			ParseMembersAndMethods(new MethodExpressionParser()).Methods[0];
}
