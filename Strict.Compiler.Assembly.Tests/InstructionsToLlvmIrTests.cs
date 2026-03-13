using NUnit.Framework;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Assembly.Tests;

public sealed class InstructionsToLlvmIrTests
{
	private readonly InstructionsToLlvmIr compiler = new();
	private static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);

	[TestCase("LlvmAdd", "+", "fadd")]
	[TestCase("LlvmSubtract", "-", "fsub")]
	[TestCase("LlvmMultiply", "*", "fmul")]
	public void GenerateArithmeticFunction(string typeName, string op, string llvmOp)
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 3.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 7.0)),
			new BinaryInstruction(op == "+" ? InstructionType.Add
				: op == "-" ? InstructionType.Subtract
				: InstructionType.Multiply, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var ir = compiler.CompileInstructions(typeName, instructions);
		Assert.That(ir, Does.Contain($"define double @{typeName}("));
		Assert.That(ir, Does.Contain(llvmOp + " double"));
		Assert.That(ir, Does.Contain("ret double"));
	}

	[Test]
	public void GenerateDivideFunction()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 10.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 2.0)),
			new BinaryInstruction(InstructionType.Divide, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var ir = compiler.CompileInstructions("Divide", instructions);
		Assert.That(ir, Does.Contain("fdiv double"));
	}

	[Test]
	public void FunctionContainsDefineAndRet()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileInstructions("Run", instructions);
		Assert.That(ir, Does.Contain("define double @Run("));
		Assert.That(ir, Does.Contain("ret double"));
	}

	[Test]
	public void CompileInstructionsUsesGivenMethodName()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileInstructions("MyMethod", instructions);
		Assert.That(ir, Does.Contain("define double @MyMethod("));
	}

	[Test]
	public void ZeroConstantUsesLiteralZero()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileInstructions("ZeroTest", instructions);
		Assert.That(ir, Does.Contain("ret double 0.0"));
	}

	[Test]
	public void LocalVariableUsesAllocaAndStore()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 3.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 7.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new StoreFromRegisterInstruction(Register.R2, "result"),
			new LoadVariableToRegister(Register.R3, "result"),
			new ReturnInstruction(Register.R3)
		};
		var ir = compiler.CompileInstructions("LocalVar", instructions);
		Assert.That(ir, Does.Contain("alloca double"));
		Assert.That(ir, Does.Contain("store double"));
		Assert.That(ir, Does.Contain("load double"));
	}

	[Test]
	public void ConditionalBranchGeneratesFcmp()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 0.0)),
			new BinaryInstruction(InstructionType.GreaterThan, Register.R0, Register.R1),
			new Jump(1, InstructionType.JumpIfFalse),
			new ReturnInstruction(Register.R0),
			new ReturnInstruction(Register.R1)
		};
		var ir = compiler.CompileInstructions("Conditional", instructions);
		Assert.That(ir, Does.Contain("fcmp ogt double"));
		Assert.That(ir, Does.Contain("br i1"));
	}

	[Test]
	public void UnconditionalJumpGeneratesBrLabel()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new Jump(1),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 99.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileInstructions("UncondJump", instructions);
		Assert.That(ir, Does.Contain("br label %L"));
	}

	[Test]
	public void CompileWindowsPlatformIncludesMainEntryPoint()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.Windows);
		Assert.That(ir, Does.Contain("define i32 @main()"));
		Assert.That(ir, Does.Contain("call double @Run()"));
		Assert.That(ir, Does.Contain("ret i32 0"));
	}

	[Test]
	public void CompileLinuxPlatformIncludesTargetTriple()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		Assert.That(ir, Does.Contain("target triple = \"x86_64-unknown-linux-gnu\""));
		Assert.That(ir, Does.Contain("define i32 @main()"));
	}

	[Test]
	public void CompileMacOsPlatformIncludesTargetTriple()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.MacOS);
		Assert.That(ir, Does.Contain("target triple = \"x86_64-apple-macosx\""));
	}

	[Test]
	public void FunctionBodyAppearsBeforeMainEntryPoint()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Work", instructions, Platform.Linux);
		var funcPos = ir.IndexOf("define double @Work(", StringComparison.Ordinal);
		var mainPos = ir.IndexOf("define i32 @main()", StringComparison.Ordinal);
		Assert.That(funcPos, Is.LessThan(mainPos),
			"Function body should appear before main entry point");
	}

	[Test]
	public void LlvmIrDoesNotContainPlatformSpecificAssembly()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var ir = compiler.CompileForPlatform("Compute", instructions, Platform.Linux);
		Assert.That(ir, Does.Not.Contain("xmm"));
		Assert.That(ir, Does.Not.Contain("movsd"));
		Assert.That(ir, Does.Not.Contain("push rbp"));
		Assert.That(ir, Does.Not.Contain("section .text"));
	}

	[Test]
	public void LlvmIrIsMuchShorterThanNasmForSameInstructions()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var llvmIr = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		var nasmAsm = new InstructionsToAssembly().CompileForPlatform("Run", instructions,
			Platform.Linux);
		Assert.That(llvmIr.Length, Is.LessThan(nasmAsm.Length),
			"LLVM IR should be shorter than NASM assembly for the same program");
	}

	[Test]
	public void CompileForPlatformSupportsInvokeWithPrecompiledMethodBytecode()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("LlvmInvokeType",
			"has dummy Number",
			"Add(first Number, second Number) Number",
			"\tfirst + second",
			"Run Number",
			"\tAdd(2, 3)")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = new BytecodeGenerator(new InvokedMethod(
			(addMethod.GetBodyAndParseIfNeeded() as Body)?.Expressions ??
			[addMethod.GetBodyAndParseIfNeeded()],
			new Dictionary<string, ValueInstance>(), addMethod.ReturnType), new Registry()).Generate();
		var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(type.Name, addMethod.Name,
			addMethod.Parameters.Count);
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Windows,
			new Dictionary<string, List<Instruction>> { [methodKey] = addInstructions });
		Assert.That(ir, Does.Contain("call double @" + type.Name + "_Add_2("));
	}

	[Test]
	public void HasPrintInstructionsReturnsTrueForPrintInstructions()
	{
		var instructions = new List<Instruction>
		{
			new PrintInstruction("Hello"),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		Assert.That(compiler.HasPrintInstructions(instructions), Is.True);
	}

	[Test]
	public void HasPrintInstructionsReturnsFalseWithoutPrint()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		Assert.That(compiler.HasPrintInstructions(instructions), Is.False);
	}

	[Test]
	public void PrintInstructionDeclaresprintf()
	{
		var instructions = new List<Instruction>
		{
			new PrintInstruction("Hello"),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		Assert.That(ir, Does.Contain("declare i32 @printf(ptr, ...)"));
		Assert.That(ir, Does.Contain("call i32 (ptr, ...) @printf("));
	}

	[Test]
	public void LlvmLinkerIsClangAvailableReturnsBooleanWithoutThrowing() =>
		Assert.That(LlvmLinker.IsClangAvailable, Is.TypeOf<bool>());

	[Test]
	public void LlvmLinkerThrowsToolNotFoundWhenClangMissing()
	{
		if (LlvmLinker.IsClangAvailable)
			return;
		var linker = new LlvmLinker();
		var tempLl = Path.Combine(Path.GetTempPath(), "test_" + Guid.NewGuid() + ".ll");
		File.WriteAllText(tempLl,
			"define i32 @main() { ret i32 0 }");
		try
		{
			Assert.Throws<ToolNotFoundException>(() =>
				linker.CreateExecutable(tempLl, Platform.Windows));
		}
		finally
		{
			if (File.Exists(tempLl))
				File.Delete(tempLl);
		}
	}

	[Test]
	public void ToolNotFoundExceptionContainsClangAndUrl()
	{
		var exception = new ToolNotFoundException("clang", "https://releases.llvm.org");
		Assert.That(exception.Message, Does.Contain("clang"));
		Assert.That(exception.Message, Does.Contain("https://releases.llvm.org"));
	}

	[Test]
	public void CompileForPlatformSupportsConstructorAndInstanceMethodCalls()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("LlvmSimpleCalc",
			"has first Number",
			"has second Number",
			"Add Number",
			"\tfirst + second",
			"Run Number",
			"\tconstant calc = LlvmSimpleCalc(2, 3)",
			"\tcalc.Add")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = new BytecodeGenerator(new InvokedMethod(
			(addMethod.GetBodyAndParseIfNeeded() as Body)?.Expressions ??
			[addMethod.GetBodyAndParseIfNeeded()],
			new Dictionary<string, ValueInstance>(), addMethod.ReturnType), new Registry()).Generate();
		var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(type.Name, addMethod.Name,
			addMethod.Parameters.Count);
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux,
			new Dictionary<string, List<Instruction>> { [methodKey] = addInstructions });
		Assert.That(ir, Does.Contain("define double @" + type.Name + "_Add_0("));
		Assert.That(ir, Does.Contain("call double @" + type.Name + "_Add_0("));
	}
}
