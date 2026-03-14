using NUnit.Framework;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Assembly.Tests;

public sealed class InstructionsToAssemblyTests
{
	private readonly InstructionsToAssembly compiler = new();
	private static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);

	[TestCase("ArithAdd", "+", "addsd")]
	[TestCase("ArithSubtract", "-", "subsd")]
	[TestCase("ArithMultiply", "*", "mulsd")]
	public void GenerateArithmeticFunction(string typeName, string op, string asmOp)
	{
		var method = CreateSingleMethod(typeName, "has dummy Number",
			$"{typeName}(first Number, second Number) Number", $"\tfirst {op} second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain($"global {typeName}"));
		Assert.That(assembly, Does.Contain($"{typeName}:"));
		Assert.That(assembly, Does.Contain(asmOp));
		Assert.That(assembly, Does.Contain("ret"));
	}

	[Test]
	public void GenerateDivideFunction()
	{
		var method = CreateSingleMethod("DivideType", "has dummy Number",
			"Divide(numerator Number, denominator Number) Number", "\tnumerator / denominator");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("divsd"));
		Assert.That(assembly, Does.Contain("ret"));
	}

	[Test]
	public void FunctionHasTextSectionAndEpilogue()
	{
		var method = CreateSingleMethod("FunctionFrameType", "has dummy Number",
			"Sum(first Number, second Number) Number", "\tlet result = first + second",
			"\tresult + 1");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("section .text"));
		Assert.That(assembly, Does.Contain("push rbp"));
		Assert.That(assembly, Does.Contain("mov rbp, rsp"));
		Assert.That(assembly, Does.Contain("pop rbp"));
		Assert.That(assembly, Does.Contain("ret"));
	}

	[Test]
	public void LeafFunctionHasTextSectionAndRet()
	{
		var method = CreateSingleMethod("SumLeafType", "has dummy Number",
			"Sum(first Number, second Number) Number", "\tfirst + second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("section .text"));
		Assert.That(assembly, Does.Not.Contain("push rbp"));
		Assert.That(assembly, Does.Not.Contain("mov rbp, rsp"));
		Assert.That(assembly, Does.Not.Contain("pop rbp"));
		Assert.That(assembly, Does.Contain("ret"));
	}

	[Test]
	public void FunctionWithLocalVariableHasTextSectionAndEpilogue()
	{
		var method = CreateSingleMethod("LocalFrameType", "has dummy Number",
			"Sum(first Number, second Number) Number", "\tlet result = first + second",
			"\tresult + 1");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("section .text"));
		Assert.That(assembly, Does.Contain("push rbp"));
		Assert.That(assembly, Does.Contain("mov rbp, rsp"));
		Assert.That(assembly, Does.Contain("pop rbp"));
		Assert.That(assembly, Does.Contain("ret"));
	}

	[Test]
	public void ParametersStayInRegistersForLeafMethods()
	{
		var method = CreateSingleMethod("TotalType", "has dummy Number",
			"Total(first Number, second Number) Number", "\tfirst + second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Not.Contain("movsd [rbp-8], xmm0"));
		Assert.That(assembly, Does.Not.Contain("movsd [rbp-16], xmm1"));
	}

	[Test]
	public void ReturnValueCanBeProducedDirectlyInXmm0()
	{
		var method = CreateSingleMethod("SumNumbers", "has dummy Number",
			"SumTwo(first Number, second Number) Number", "\tfirst + second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("addsd xmm0, xmm1"));
		Assert.That(assembly, Does.Not.Contain("movsd xmm0, xmm2"));
	}

	[Test]
	public void NonZeroConstantGoesToDataSection()
	{
		var method = CreateSingleMethod("AddFiveType", "has dummy Number",
			"AddFive(first Number, second Number) Number", "\tfirst + 5");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("section .data"));
		Assert.That(assembly, Does.Contain("dq 0x"));
		Assert.That(assembly, Does.Contain("[rel const_"));
	}

	[Test]
	public void ZeroConstantUsesXorpd()
	{
		var method = CreateSingleMethod("ZeroAddType", "has dummy Number",
			"ZeroAdd(first Number, second Number) Number", "\tfirst + 0");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("xorpd"));
	}

	[Test]
	public void LocalVariableIsStoredAndLoadedFromStack()
	{
		var method = CreateSingleMethod("CalcType", "has dummy Number",
			"Calculate(first Number, second Number) Number", "\tlet result = first + second",
			"\tresult + 1");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("movsd [rbp-"));
		Assert.That(assembly, Does.Contain("movsd xmm"));
		Assert.That(assembly, Does.Contain("addsd"));
	}

	[Test]
	public void CompileInstructionsFromListUsesGivenMethodName()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileInstructions("Run", instructions);
		Assert.That(assembly, Does.Contain("global Run"));
		Assert.That(assembly, Does.Contain("Run:"));
		Assert.That(assembly, Does.Contain("section .text"));
		Assert.That(assembly, Does.Contain("ret"));
	}

	[Test]
	public void CompileInstructionsHasNoPrologueParamSpillsForEmptyParamList()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileInstructions("Run", instructions);
		Assert.That(assembly, Does.Not.Contain("movsd [rbp-8], xmm0"));
	}

	[Test]
	public void SingleParameterFunctionUsesRegisterOnly()
	{
		var method = CreateSingleMethod("DoubleType", "has dummy Number",
			"DoubleNum(num Number) Number", "\tnum + num");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Not.Contain("movsd [rbp-8], xmm0"));
		Assert.That(assembly, Does.Not.Contain("movsd [rbp-16], xmm1"));
	}

	[Test]
	public void TwoParameterLeafMethodNeedsNoStackFrame()
	{
		var method = CreateSingleMethod("TwoParmType", "has dummy Number",
			"TwoParam(first Number, second Number) Number", "\tfirst + second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Not.Contain("sub rsp,"));
	}

	[Test]
	public void ConditionalBranchGeneratesUcomisd()
	{
		var method = CreateSingleMethod("ConditionalType", "has dummy Number",
			"IsPositive(num Number) Number", "\tif num > 0", "\t\treturn num", "\t0");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("ucomisd"));
	}

	[Test]
	public void ConditionalBranchGeneratesJbeForGreaterThanJumpIfFalse()
	{
		var method = CreateSingleMethod("GreaterThanType", "has dummy Number",
			"IsAboveZero(num Number) Number", "\tif num > 0", "\t\treturn num", "\t0");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("jbe"));
	}

	[Test]
	public void EqualComparisonJumpIfFalseGeneratesJne()
	{
		var method = CreateSingleMethod("EqualJumpType", "has dummy Number",
			"IsEqualFive(num Number) Number", "\tif num is 5", "\t\treturn num", "\t0");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("jne"));
	}

	[Test]
	public void JumpIfFalseGeneratesJne()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 2.0)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new Jump(2, InstructionType.JumpIfFalse),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 10.0)),
			new ReturnInstruction(Register.R0),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileInstructions("TestJump", instructions);
		Assert.That(assembly, Does.Contain("jne"));
	}

	[Test]
	public void JumpIfTrueGeneratesJe()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 2.0)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new Jump(2, InstructionType.JumpIfTrue),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 10.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileInstructions("TestJumpTrue", instructions);
		Assert.That(assembly, Does.Contain("je"));
	}

	[Test]
	public void UnconditionalJumpGeneratesJmp()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new Jump(1),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 99.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileInstructions("TestUncondJump", instructions);
		Assert.That(assembly, Does.Contain("jmp"));
	}

	[Test]
	public void JumpCreatesLabelAtTargetInstruction()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new Jump(1),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 99.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileInstructions("TestLabel", instructions);
		Assert.That(assembly, Does.Contain(".L0:"));
		var lines = assembly.Split('\n');
		var labelLine = Array.FindIndex(lines, l => l.Contains(".L0:"));
		var retLine = Array.FindIndex(lines,
			l => l.TrimStart().StartsWith("ret", StringComparison.Ordinal));
		Assert.That(labelLine, Is.LessThan(retLine));
	}

	[Test]
	public void SameConstantUsedTwiceCreatesOnlyOneDataEntry()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 7.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 7.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var assembly = compiler.CompileInstructions("Dedup", instructions);
		Assert.That(assembly.Split('\n').Count(l => l.Contains("const_")), Is.EqualTo(3));
	}

	[Test]
	public void TwoDifferentConstantsBothAppearInDataSection()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 3.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 7.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var assembly = compiler.CompileInstructions("TwoConsts", instructions);
		Assert.That(assembly, Does.Contain("const_0"));
		Assert.That(assembly, Does.Contain("const_1"));
	}

	[Test]
	public void RegisterMappingIsCorrectForHighRegisters()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R5, new ValueInstance(NumberType, 1.0)),
			new LoadConstantInstruction(Register.R10, new ValueInstance(NumberType, 2.0)),
			new BinaryInstruction(InstructionType.Add, Register.R5, Register.R10, Register.R15),
			new ReturnInstruction(Register.R15)
		};
		var assembly = compiler.CompileInstructions("HighRegs", instructions);
		Assert.That(assembly, Does.Contain("xmm5"));
		Assert.That(assembly, Does.Contain("xmm10"));
		Assert.That(assembly, Does.Contain("xmm15"));
	}

	[Test]
	public void DataSectionAppearsBeforeTextSection()
	{
		var method = CreateSingleMethod("OrderTestType", "has dummy Number",
			"OrderTest(first Number, second Number) Number", "\tfirst + 9");
		var assembly = compiler.Compile(method);
		var dataPos = assembly.IndexOf("section .data", StringComparison.Ordinal);
		var textPos = assembly.IndexOf("section .text", StringComparison.Ordinal);
		Assert.That(dataPos, Is.LessThan(textPos));
	}

	[Test]
	public void SubtractionUsesSubsdInstruction()
	{
		var method = CreateSingleMethod("SubOrderType", "has dummy Number",
			"SubOrder(first Number, second Number) Number", "\tfirst - second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("subsd"));
		Assert.That(assembly, Does.Contain("xmm0"));
		Assert.That(assembly, Does.Contain("xmm1"));
	}

	[Test]
	public void JumpToIdIfFalseWithEqualComparisonGeneratesJne()
	{
		var method = CreateSingleMethod("JumpToIdType", "has dummy Number",
			"JumpToId(operation Text) Number", "\tif operation is \"add\"", "\t\treturn 1", "\t0");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("jne"));
		Assert.That(assembly, Does.Contain(".L"));
	}

	[Test]
	public void CompileWindowsPlatformIncludesMainEntryPoint()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileForPlatform("Run", instructions, Platform.Windows);
		Assert.That(assembly, Does.Contain("global main"));
		Assert.That(assembly, Does.Contain("main:"));
		Assert.That(assembly, Does.Contain("call Run"));
	}

	[Test]
	public void CompileWindowsPlatformIncludesExitProcess()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileForPlatform("Compute", instructions, Platform.Windows);
		Assert.That(assembly, Does.Contain("extern ExitProcess"));
		Assert.That(assembly, Does.Contain("call ExitProcess"));
		Assert.That(assembly, Does.Contain("xor rcx, rcx"));
	}

	[Test]
	public void CompileWindowsPlatformContainsBothFunctionAndEntryPoint()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileForPlatform("MyFunc", instructions, Platform.Windows);
		Assert.That(assembly, Does.Contain("global MyFunc"));
		Assert.That(assembly, Does.Contain("MyFunc:"));
		Assert.That(assembly, Does.Contain("global main"));
		Assert.That(assembly, Does.Contain("call MyFunc"));
	}

	[Test]
	public void CompileWindowsPlatformEntryPointAppearsAfterFunction()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileForPlatform("Work", instructions, Platform.Windows);
		var funcPos = assembly.IndexOf("Work:", StringComparison.Ordinal);
		var mainPos = assembly.IndexOf("main:", StringComparison.Ordinal);
		Assert.That(funcPos, Is.LessThan(mainPos),
			"Function body should appear before main entry point");
	}

	[Test]
	public void CompileWindowsPlatformHasShadowSpaceForWindowsAbi()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileForPlatform("Entry", instructions, Platform.Windows);
		Assert.That(assembly, Does.Contain("sub rsp, 32"),
			"Windows ABI requires 32-byte shadow space");
		Assert.That(assembly, Does.Contain("add rsp, 32"));
	}

	[Test]
	public void CompileLinuxPlatformUsesStartEntryPointAndSyscallExit()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		Assert.That(assembly, Does.Contain("global _start"));
		Assert.That(assembly, Does.Contain("_start:"));
		Assert.That(assembly, Does.Contain("call Run"));
		Assert.That(assembly, Does.Contain("mov rax, 60"));
		Assert.That(assembly, Does.Contain("syscall"));
	}

	[Test]
	public void CompileMacOsPlatformUsesMainAndSyscallExit()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileForPlatform("Run", instructions, Platform.MacOS);
		Assert.That(assembly, Does.Contain("global _main"));
		Assert.That(assembly, Does.Contain("_main:"));
		Assert.That(assembly, Does.Contain("0x2000001"));
		Assert.That(assembly, Does.Contain("syscall"));
	}

	[Test]
	public void NativeExecutableLinkerThrowsToolNotFoundWhenNasmMissing()
	{
		if (NativeExecutableLinker.IsNasmAvailable)
			return; //ncrunch: no coverage start, only executed when nasm is missing
		var linker = new NativeExecutableLinker();
		var tempAsm = Path.Combine(Path.GetTempPath(), "test_" + Guid.NewGuid() + ".asm");
		File.WriteAllText(tempAsm, "section .text\nglobal main\nmain:\n    ret");
		try
		{
			Assert.Throws<ToolNotFoundException>(() =>
				linker.CreateExecutable(tempAsm, Platform.Windows));
		}
		finally
		{
			if (File.Exists(tempAsm))
				File.Delete(tempAsm);
		}
	} //ncrunch: no coverage end

	[Test]
	public void NativeExecutableLinkerIsNasmAvailableReturnsBooleanWithoutThrowing() =>
		Assert.That(NativeExecutableLinker.IsNasmAvailable, Is.TypeOf<bool>());

	[Test]
	public void NativeExecutableLinkerIsGccAvailableReturnsBooleanWithoutThrowing() =>
		Assert.That(NativeExecutableLinker.IsGccAvailable, Is.TypeOf<bool>());

	[Test]
	public void ToolNotFoundExceptionContainsNameAndUrl()
	{
		var exception = new ToolNotFoundException("nasm", "https://nasm.us");
		Assert.That(exception.Message, Does.Contain("nasm"));
		Assert.That(exception.Message, Does.Contain("https://nasm.us"));
	}

	[Test]
	public void CompileForPlatformSupportsInvokeWithPrecompiledMethodBytecode()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("InvokeCompileType",
			"has dummy Number",
			"Add(first Number, second Number) Number",
			"\tfirst + second",
			"Run Number",
			"\tAdd(2, 3)")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = new BytecodeGenerator(new InvokedMethod(
			(addMethod.GetBodyAndParseIfNeeded() as Body)?.Expressions ?? [addMethod.GetBodyAndParseIfNeeded()],
			new Dictionary<string, ValueInstance>(), addMethod.ReturnType), new Registry()).Generate();
		var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(type.Name, addMethod.Name,
			addMethod.Parameters.Count);
		var assembly = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Windows,
			new Dictionary<string, List<Instruction>> { [methodKey] = addInstructions });
		Assert.That(assembly, Does.Contain("call " + type.Name + "_Add_2"));
	}

	[Test]
	public void CompileForPlatformSupportsSimpleCalculatorStyleConstructorAndInstanceMethodCalls()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("AssemblySimpleCalculator",
			"has first Number",
			"has second Number",
			"Add Number",
			"\tfirst + second",
			"Multiply Number",
			"\tfirst * second",
			"Run Number",
			"\tconstant calc = AssemblySimpleCalculator(2, 3)",
			"\tconstant added = calc.Add",
			"\tconstant multiplied = calc.Multiply",
			"\tadded + multiplied")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var multiplyMethod = type.Methods.First(method => method.Name == "Multiply");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = new BytecodeGenerator(new InvokedMethod(
			(addMethod.GetBodyAndParseIfNeeded() as Body)?.Expressions ?? [addMethod.GetBodyAndParseIfNeeded()],
			new Dictionary<string, ValueInstance>(), addMethod.ReturnType), new Registry()).Generate();
		var multiplyInstructions = new BytecodeGenerator(new InvokedMethod(
			(multiplyMethod.GetBodyAndParseIfNeeded() as Body)?.Expressions ?? [multiplyMethod.GetBodyAndParseIfNeeded()],
			new Dictionary<string, ValueInstance>(), multiplyMethod.ReturnType), new Registry()).Generate();
		var addMethodKey = BytecodeDeserializer.BuildMethodInstructionKey(type.Name, addMethod.Name,
			addMethod.Parameters.Count);
		var multiplyMethodKey = BytecodeDeserializer.BuildMethodInstructionKey(type.Name,
			multiplyMethod.Name, multiplyMethod.Parameters.Count);
		var assembly = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux,
			new Dictionary<string, List<Instruction>>
			{
				[addMethodKey] = addInstructions,
				[multiplyMethodKey] = multiplyInstructions
			});
		Assert.That(assembly, Does.Contain(type.Name + "_Add_0:"));
		Assert.That(assembly, Does.Contain(type.Name + "_Multiply_0:"));
		var addLabel = type.Name + "_Add_0:";
		var addStart = assembly.IndexOf(addLabel, StringComparison.Ordinal);
		var addEnd = assembly.IndexOf("section .text", addStart + addLabel.Length, StringComparison.Ordinal);
		var addBody = assembly.Substring(addStart, addEnd - addStart);
		Assert.That(addBody, Does.Not.Contain("movsd [rbp-"));
		var multiplyLabel = type.Name + "_Multiply_0:";
		var multiplyStart = assembly.IndexOf(multiplyLabel, StringComparison.Ordinal);
		var multiplyEnd = assembly.IndexOf("global _start", multiplyStart + multiplyLabel.Length,
			StringComparison.Ordinal);
		var multiplyBody = assembly.Substring(multiplyStart, multiplyEnd - multiplyStart);
		Assert.That(multiplyBody, Does.Not.Contain("movsd [rbp-"));
	}

	private static Method CreateSingleMethod(string typeName, params string[] methodLines) =>
		new Type(TestPackage.Instance, new TypeLines(typeName, methodLines)).
			ParseMembersAndMethods(new MethodExpressionParser()).Methods[0];

	[Test]
	public void LeafAddWithTwoParametersUsesDirectRegistersWithoutStackSpills()
	{
		var method = CreateSingleMethod("LeafAddType", "has dummy Number",
			"Add(first Number, second Number) Number", "\tfirst + second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("addsd xmm0, xmm1"));
		Assert.That(assembly, Does.Not.Contain("movsd [rbp-8], xmm0"));
		Assert.That(assembly, Does.Not.Contain("movsd [rbp-16], xmm1"));
	}

	[Test]
	public void LeafMultiplyWithTwoParametersUsesDirectRegistersWithoutStackSpills()
	{
		var method = CreateSingleMethod("LeafMultiplyType", "has dummy Number",
			"Multiply(first Number, second Number) Number", "\tfirst * second");
		var assembly = compiler.Compile(method);
		Assert.That(assembly, Does.Contain("mulsd xmm0, xmm1"));
		Assert.That(assembly, Does.Not.Contain("movsd [rbp-8], xmm0"));
		Assert.That(assembly, Does.Not.Contain("movsd [rbp-16], xmm1"));
	}

	[Test]
	public void ConstantLeafFunctionNeedsNoFrameInstructions()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new ReturnInstruction(Register.R0)
		};
		var assembly = compiler.CompileInstructions("Run", instructions);
		Assert.That(assembly, Does.Not.Contain("push rbp"));
		Assert.That(assembly, Does.Not.Contain("mov rbp, rsp"));
		Assert.That(assembly, Does.Not.Contain("pop rbp"));
	}

	[Test]
	public void PlatformCompiledMemberCallsDoNotEmitDeadXmmInitialization()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("DeadRegisterInitType",
			"has first Number",
			"has second Number",
			"Add Number",
			"\tfirst + second",
			"Run Number",
			"\tconstant calc = DeadRegisterInitType(2, 3)",
			"\tcalc.Add")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = new BytecodeGenerator(new InvokedMethod(
			(addMethod.GetBodyAndParseIfNeeded() as Body)?.Expressions ?? [addMethod.GetBodyAndParseIfNeeded()],
			new Dictionary<string, ValueInstance>(), addMethod.ReturnType), new Registry()).Generate();
		var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(type.Name, addMethod.Name,
			addMethod.Parameters.Count);
		var assembly = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux,
			new Dictionary<string, List<Instruction>> { [methodKey] = addInstructions });
		Assert.That(assembly, Does.Not.Contain("xorpd xmm2, xmm2"));
	}

	[Test]
	public void TwoNumericWindowsPrintsReuseSharedPrintNumberHelper()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new PrintInstruction("A = ", Register.R0),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 6.0)),
			new PrintInstruction("B = ", Register.R1),
			new ReturnInstruction(Register.R1)
		};
		var assembly = compiler.CompileForPlatform("Run", instructions, Platform.Windows);
		Assert.That(assembly, Does.Contain("print_number_from_xmm:"));
		Assert.That(assembly.Split("call print_number_from_xmm").Length - 1,
			Is.EqualTo(2));
		Assert.That(assembly.Split("cvttsd2si rax, xmm0").Length - 1,
			Is.EqualTo(1));
	}

	[Test]
	public void LinuxOutputCheckOnWindowsAcceptsExeFallback()
	{
		if (!OperatingSystem.IsWindows())
			return; //ncrunch: no coverage
		var tempDirectoryPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(tempDirectoryPath);
		var expectedLinuxOutputPath = Path.Combine(tempDirectoryPath, "PureAdder");
		File.WriteAllText(expectedLinuxOutputPath + ".exe", "probe");
		Assert.DoesNotThrow(() =>
			ToolRunner.EnsureOutputFileExists(expectedLinuxOutputPath, "gcc", Platform.Linux));
	}

	[Test]
	public void MissingNativeOutputFileThrowsDetailedInvalidOperationException()
	{
		var missingFilePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "PureAdder");
		var exception = Assert.Throws<InvalidOperationException>(() =>
			ToolRunner.EnsureOutputFileExists(missingFilePath, "gcc", Platform.Linux))!;
		Assert.That(exception.Message, Does.Contain(missingFilePath));
		Assert.That(exception.Message, Does.Contain("gcc"));
	}
}