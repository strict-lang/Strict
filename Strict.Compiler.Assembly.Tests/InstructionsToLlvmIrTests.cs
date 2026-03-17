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
			new BinaryInstruction(op switch //ncrunch: no coverage
			{
				"+" => InstructionType.Add,
				"-" => InstructionType.Subtract,
				_ => InstructionType.Multiply
			}, Register.R0, Register.R1, Register.R2),
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
	public void LlvmIrContainsSsaFormArithmetic()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		Assert.That(ir, Does.Contain("fadd double"));
		Assert.That(ir, Does.Contain("ret double %t"));
		Assert.That(ir, Does.Contain("define double @Run()"));
	}

	[Test]
	public void CompileForPlatformSupportsInvokeWithPrecompiledMethodBytecode()
	{
		var type =
			new Type(TestPackage.Instance,
				new TypeLines("LlvmInvokeType", "has dummy Number",
					"Add(first Number, second Number) Number", "\tfirst + second", "Run Number",
					"\tAdd(2, 3)")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var runInstructions = new BinaryGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = GenerateMethodInstructions(addMethod);
		var methodKey = BuildMethodKey(addMethod);
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Windows,
			new Dictionary<string, List<Instruction>> { [methodKey] = addInstructions });
		Assert.That(ir, Does.Contain("call double @" + type.Name + "_Add_2("));
	}

	//TODO: move to BinaryMethodTests
	[Test]
	public void HasPrintInstructionsReturnsTrueForPrintInstructions()
	{
		var instructions = new List<Instruction>
		{
			new PrintInstruction("Hello"),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		Assert.That(InstructionsToLlvmIr.HasPrintInstructions(instructions), Is.True);
	}

	[Test]
	public void HasPrintInstructionsReturnsFalseWithoutPrint()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		Assert.That(InstructionsToLlvmIr.HasPrintInstructions(instructions), Is.False);
	}

	[Test]
	public void PrintInstructionDeclaressprintfAndUsesGep()
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
		Assert.That(ir, Does.Contain("getelementptr inbounds"));
	}

	[Test]
	public void StringConstantsAreNullTerminated()
	{
		var instructions = new List<Instruction>
		{
			new PrintInstruction("Test"),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		Assert.That(ir, Does.Contain("\\00\""),
			"String constants must be null-terminated for C printf compatibility");
	}

	[Test]
	public void IsClangAvailableDoesNotThrow() =>
		Assert.DoesNotThrow(() => _ = LlvmLinker.IsClangAvailable);

	[Test]
	public void LlvmLinkerThrowsToolNotFoundWhenClangMissing()
	{
		if (LlvmLinker.IsClangAvailable)
			return;
		//ncrunch: no coverage start
		var linker = new LlvmLinker();
		var tempLl = Path.Combine(Path.GetTempPath(), "test_" + Guid.NewGuid() + ".ll");
		File.WriteAllText(tempLl, "define i32 @main() { ret i32 0 }");
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
	} //ncrunch: no coverage end

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
		var type = new Type(TestPackage.Instance,
			new TypeLines("LlvmSimpleCalc", "has first Number", "has second Number", "Add Number",
				"\tfirst + second", "Run Number", "\tconstant calc = LlvmSimpleCalc(2, 3)",
				"\tcalc.Add")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var runInstructions = new BinaryGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = GenerateMethodInstructions(addMethod);
		var methodKey = BuildMethodKey(addMethod);
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Windows,
			new Dictionary<string, List<Instruction>> { [methodKey] = addInstructions });
		Assert.That(ir, Does.Contain("define double @" + type.Name + "_Add_0("));
		Assert.That(ir, Does.Contain("call double @" + type.Name + "_Add_0("));
	}

	[Test]
	public void UnhandledInstructionThrowsNotSupportedException()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new SetInstruction(new ValueInstance(NumberType, 0.0), Register.R1),
			new ReturnInstruction(Register.R0)
		};
		Assert.Throws<NotSupportedException>(() =>
			compiler.CompileInstructions("BadInstr", instructions));
	}

	[Test]
	public void ModuloArithmeticGeneratesFrem()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 15.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.Modulo, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var ir = compiler.CompileInstructions("ModTest", instructions);
		Assert.That(ir, Does.Contain("frem double"));
	}

	[Test]
	public void NotEqualComparisonGeneratesOne()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.NotEqual, Register.R0, Register.R1),
			new Jump(1, InstructionType.JumpIfFalse),
			new ReturnInstruction(Register.R0),
			new ReturnInstruction(Register.R1)
		};
		var ir = compiler.CompileInstructions("NotEq", instructions);
		Assert.That(ir, Does.Contain("fcmp one double"));
	}

	[Test]
	public void LessThanComparisonGeneratesOlt()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 2.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 5.0)),
			new BinaryInstruction(InstructionType.LessThan, Register.R0, Register.R1),
			new Jump(1, InstructionType.JumpIfTrue),
			new ReturnInstruction(Register.R1),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileInstructions("LessThan", instructions);
		Assert.That(ir, Does.Contain("fcmp olt double"));
	}

	[Test]
	public void PureAdderStyleTypeGeneratesIrWithReturnConstant()
	{
		var type =
			new Type(TestPackage.Instance,
					new TypeLines("LlvmPureAdder", "has first Number", "has second Number", "Add Number",
						"\tfirst + second", "Run Number", "\t42")).
				ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var runInstructions = new BinaryGenerator(new MethodCall(runMethod)).Generate();
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux);
		Assert.That(ir, Does.Contain("define double @LlvmPureAdder("));
		Assert.That(ir, Does.Contain("ret double 42.0"));
		Assert.That(ir, Does.Contain("define i32 @main()"));
	}

	[Test]
	public void SimpleCalculatorStyleTypeGeneratesAddAndMultiplyFunctions()
	{
		var type = new Type(TestPackage.Instance,
				new TypeLines("LlvmCalc", "has first Number", "has second Number", "Add Number",
					"\tfirst + second", "Multiply Number", "\tfirst * second", "Run Number",
					"\tconstant calc = LlvmCalc(2, 3)", "\tconstant added = calc.Add",
					"\tconstant multiplied = calc.Multiply", "\tadded + multiplied")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var multiplyMethod = type.Methods.First(method => method.Name == "Multiply");
		var runInstructions = new BinaryGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = GenerateMethodInstructions(addMethod);
		var multiplyInstructions = GenerateMethodInstructions(multiplyMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BuildMethodKey(addMethod)] = addInstructions,
			[BuildMethodKey(multiplyMethod)] = multiplyInstructions
		};
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Windows,
			precompiled);
		Assert.That(ir, Does.Contain("define double @LlvmCalc_Add_0("));
		Assert.That(ir, Does.Contain("define double @LlvmCalc_Multiply_0("));
		Assert.That(ir, Does.Contain("call double @LlvmCalc_Add_0("));
		Assert.That(ir, Does.Contain("call double @LlvmCalc_Multiply_0("));
		Assert.That(ir, Does.Contain("fadd double"));
	}

	[Test]
	public void ArithmeticFunctionStyleTypeWithParametersGeneratesParamSignature()
	{
		var type =
			new Type(TestPackage.Instance,
				new TypeLines("LlvmArithFunc", "has dummy Number",
					"Add(first Number, second Number) Number", "\tfirst + second", "Run Number",
					"\tAdd(10, 20)")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var runInstructions = new BinaryGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = GenerateMethodInstructions(addMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BuildMethodKey(addMethod)] = addInstructions
		};
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
		Assert.That(ir, Does.Contain("define double @LlvmArithFunc_Add_2("));
		Assert.That(ir, Does.Contain("double %param0"));
		Assert.That(ir, Does.Contain("double %param1"));
	}

	[Test]
	public void AreaCalculatorStyleTypeWithMultiplyComputation()
	{
		var type = new Type(TestPackage.Instance,
				new TypeLines("LlvmArea", "has width Number", "has height Number", "Area Number",
					"\twidth * height", "Run Number", "\tconstant rect = LlvmArea(5, 3)", "\trect.Area")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var areaMethod = type.Methods.First(method => method.Name == "Area");
		var runInstructions = new BinaryGenerator(new MethodCall(runMethod)).Generate();
		var areaInstructions = GenerateMethodInstructions(areaMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BuildMethodKey(areaMethod)] = areaInstructions
		};
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
		Assert.That(ir, Does.Contain("define double @LlvmArea_Area_0("));
		Assert.That(ir, Does.Contain("fmul double"));
		Assert.That(ir, Does.Contain("call double @LlvmArea_Area_0("));
	}

	[Test]
	public void TemperatureConverterStyleTypeWithArithmeticChain()
	{
		var type = new Type(TestPackage.Instance,
			new TypeLines("LlvmTempConv", "has celsius Number", "ToFahrenheit Number",
				"\tcelsius * 1.8 + 32", "Run Number", "\tconstant conv = LlvmTempConv(100)",
				"\tconv.ToFahrenheit")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var toFMethod = type.Methods.First(method => method.Name == "ToFahrenheit");
		var runInstructions = new BinaryGenerator(new MethodCall(runMethod)).Generate();
		var toFInstructions = GenerateMethodInstructions(toFMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BuildMethodKey(toFMethod)] = toFInstructions
		};
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.MacOS, precompiled);
		Assert.That(ir, Does.Contain("target triple = \"x86_64-apple-macosx\""));
		Assert.That(ir, Does.Contain("define double @LlvmTempConv_ToFahrenheit_0("));
		Assert.That(ir, Does.Contain("fmul double"));
		Assert.That(ir, Does.Contain("fadd double"));
	}

	[Test]
	public void NumericPrintUsesSnprintfWithSafeFormat()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new PrintInstruction("Result: ", Register.R0),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		Assert.That(ir, Does.Contain("@snprintf("));
		Assert.That(ir, Does.Contain("@str.safe_s"));
		Assert.That(ir, Does.Contain("@printf(ptr @str.safe_s, ptr"));
	}

	[Test]
	public void MultipleConstantsGenerateDistinctSsaValues()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 2.0)),
			new LoadConstantInstruction(Register.R2, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R3),
			new BinaryInstruction(InstructionType.Multiply, Register.R3, Register.R2, Register.R4),
			new ReturnInstruction(Register.R4)
		};
		var ir = compiler.CompileInstructions("Chain", instructions);
		Assert.That(ir, Does.Contain("fadd double 1.0, 2.0"));
		Assert.That(ir, Does.Contain("fmul double %t0, 3.0"));
	}

	[Test]
	public void StoreVariableIgnoresTextValues()
	{
		var instructions = new List<Instruction>
		{
			new StoreVariableInstruction(new ValueInstance("Hello"), "greeting"),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileInstructions("TextIgnore", instructions);
		Assert.That(ir, Does.Not.Contain("alloca"));
		Assert.That(ir, Does.Contain("ret double 0.0"));
	}

	[Test]
	public void EqualComparisonUsesLastConditionTemp()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 5.0)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new Jump(1, InstructionType.JumpIfFalse),
			new ReturnInstruction(Register.R0),
			new ReturnInstruction(Register.R1)
		};
		var ir = compiler.CompileInstructions("EqualTest", instructions);
		Assert.That(ir, Does.Contain("fcmp oeq double"));
		Assert.That(ir, Does.Contain("br i1 %t0"));
	}

	[Test]
	public void PixelStyleTypeWithThreeMembersAndDivide()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("LlvmPixel",
			"has red Number",
			"has green Number",
			"has blue Number",
			"Brighten Number",
			"\tred + green",
			"Darken Number",
			"\tred / 2",
			"Run Number",
			"\tconstant pixel = LlvmPixel(100, 150, 200)",
			"\tpixel.Brighten")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var brightenMethod = type.Methods.First(method => method.Name == "Brighten");
		var runInstructions = new BinaryGenerator(new MethodCall(runMethod)).Generate();
		var brightenInstructions = GenerateMethodInstructions(brightenMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BuildMethodKey(brightenMethod)] = brightenInstructions
		};
		var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
		Assert.That(ir, Does.Contain("define double @LlvmPixel_Brighten_0("));
		Assert.That(ir, Does.Contain("fadd double"));
		Assert.That(ir, Does.Contain("call double @LlvmPixel_Brighten_0("));
	}

	[Test]
	public void ToolRunnerFindToolReturnsNullForMissingTool() =>
		Assert.That(ToolRunner.FindTool("nonexistent_tool_xyz"), Is.Null);

	[Test]
	public void ToolRunnerEnsureOutputFileExistsThrowsForMissingFile()
	{
		var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".missing");
		Assert.Throws<InvalidOperationException>(() =>
			ToolRunner.EnsureOutputFileExists(path, "test", Platform.Linux));
	}

	private static string BuildMethodKey(Method method) =>
		BinaryExecutable.BuildMethodHeader(method.Name,
			method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.Name, null)).ToList(),
			method.ReturnType);

	private static List<Instruction> GenerateMethodInstructions(Method method)
	{
		var body = method.GetBodyAndParseIfNeeded();
		var expressions = body is Body b
			? b.Expressions
			: [body];
		return new BinaryGenerator(
			new InvokedMethod(expressions, new Dictionary<string, ValueInstance>(), method.ReturnType),
			new Registry()).Generate();
	}

	[Test]
	public void NumericPrintsWithDifferentOperatorsUseDistinctStringLabels()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 6.0)),
			new PrintInstruction("2 + 3 = ", Register.R0),
			new PrintInstruction("2 * 3 = ", Register.R1),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.Windows);
		var printConstantLines = ir.Split('\n').Where(line =>
			line.StartsWith("@str.", StringComparison.Ordinal) &&
			line.Contains("private unnamed_addr constant", StringComparison.Ordinal) &&
			!line.StartsWith("@str.safe_s", StringComparison.Ordinal)).ToList();
		Assert.That(printConstantLines.Count, Is.EqualTo(2));
		var printLabels = printConstantLines.Select(line => line[..line.IndexOf(' ')]).
			Distinct(StringComparer.Ordinal).ToList();
		Assert.That(printLabels.Count, Is.EqualTo(2));
	}

	[Test]
	public void LlvmLinkerWindowsNoPrintUsesKernel32WithoutCrt()
	{
		var args = GetLlvmClangArgs(Platform.Windows);
		Assert.That(args, Does.Contain("-nostdlib"));
		Assert.That(args, Does.Contain("-Wl,/ENTRY:main"));
		Assert.That(args, Does.Contain("-lkernel32"));
	}

	[Test]
	public void LlvmLinkerWindowsPrintUsesKernel32WithoutCrt()
	{
		var args = GetLlvmClangArgs(Platform.Windows, true);
		Assert.That(args, Does.Contain("-nostdlib"));
		Assert.That(args, Does.Contain("-Wl,/ENTRY:main"));
		Assert.That(args, Does.Contain("-lkernel32"));
	}

	[Test]
	public void LlvmLinkerWindowsNoPrintKeepsSizeFlags()
	{
		var args = GetLlvmClangArgs(Platform.Windows);
		Assert.That(args, Does.Contain("-Oz"));
		Assert.That(args, Does.Contain("/OPT:REF"));
		Assert.That(args, Does.Contain("/OPT:ICF"));
	}

	[TestCase(Platform.Windows, "-Oz")]
	[TestCase(Platform.Windows, "/OPT:REF")]
	[TestCase(Platform.Windows, "/OPT:ICF")]
	[TestCase(Platform.Linux, "-Oz")]
	[TestCase(Platform.MacOS, "-Oz")]
	public void LlvmLinkerUsesSizeOptimizationFlags(Platform platform, string expectedFlag)
	{
		var args = GetLlvmClangArgs(platform);
		Assert.That(args, Does.Contain(expectedFlag));
		Assert.That(args, Does.Not.Contain(" -O2 "));
	}

	private static string GetLlvmClangArgs(Platform platform, bool hasPrintCalls = false)
	{
		var method = typeof(LlvmLinker).GetMethod("BuildClangArgs",
				System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static) ??
			throw new InvalidOperationException("BuildClangArgs method not found");
		var parameters = method.GetParameters();
		var args = parameters.Length == 4
			? new object[] { "input.ll", "output.exe", platform, hasPrintCalls }
			: ["input.ll", "output.exe", platform];
		return (string)(method.Invoke(null, args) ??
			throw new InvalidOperationException("BuildClangArgs returned null"));
	}

	[Test]
	public void WindowsPrintInstructionUsesWinApiWithoutPrintf()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new PrintInstruction("Result = ", Register.R0),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.Windows);
		Assert.That(ir, Does.Contain("declare ptr @GetStdHandle(i32)"));
		Assert.That(ir, Does.Contain("declare i32 @WriteFile(ptr, ptr, i32, ptr, ptr)"));
		Assert.That(ir, Does.Contain("call ptr @GetStdHandle(i32 -11)"));
		Assert.That(ir, Does.Contain("call i32 @WriteFile("));
		Assert.That(ir, Does.Not.Contain("declare i32 @printf(ptr, ...)"));
		Assert.That(ir, Does.Not.Contain("declare i32 @snprintf(ptr, i64, ptr, ...)"));
	}

	[Test]
	public void WindowsLlvmModuleDefinesFltusedForNoCrtLinking()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.5)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileForPlatform("Run", instructions, Platform.Windows);
		Assert.That(ir, Does.Contain("@_fltused = global i32 0"));
	}

	[Test]
	public void LinuxTargetOnWindowsHostAvoidsGnuLinkerEntryFlags()
	{
		if (!OperatingSystem.IsWindows())
			return; //ncrunch: no coverage
		var args = GetLlvmClangArgs(Platform.Linux);
		Assert.That(args, Does.Not.Contain("-Wl,-e,main"));
		Assert.That(args, Does.Not.Contain("--gc-sections"));
		Assert.That(args, Does.Not.Contain("--strip-all"));
	}
}