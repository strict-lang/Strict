using System.Reflection;
using NUnit.Framework;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Assembly.Tests;

public sealed class InstructionsToMlirTests
{
	private readonly InstructionsToMlir compiler = new();
	private static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);

	[TestCase("MlirAdd", "+", "arith.addf")]
	[TestCase("MlirSubtract", "-", "arith.subf")]
	[TestCase("MlirMultiply", "*", "arith.mulf")]
	public void GenerateArithmeticFunction(string typeName, string op, string mlirOp)
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
		Assert.That(ir, Does.Contain($"func.func @{typeName}("));
		Assert.That(ir, Does.Contain(mlirOp));
		Assert.That(ir, Does.Contain("return"));
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
		Assert.That(ir, Does.Contain("arith.divf"));
	}

	[Test]
	public void ModuloGeneratesRemf()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 15.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.Modulo, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var ir = compiler.CompileInstructions("ModTest", instructions);
		Assert.That(ir, Does.Contain("arith.remf"));
	}

	[Test]
	public void FunctionContainsFuncFuncAndReturn()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileInstructions("Run", instructions);
		Assert.That(ir, Does.Contain("func.func @Run("));
		Assert.That(ir, Does.Contain("return"));
		Assert.That(ir, Does.Contain("f64"));
	}

	[Test]
	public void ZeroConstantUsesArithConstantZero()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 0.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileInstructions("ZeroTest", instructions);
		Assert.That(ir, Does.Contain("arith.constant 0.0 : f64"));
	}

	[Test]
	public void LocalVariableUsesDirectSsaValuePropagation()
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
		Assert.That(ir, Does.Contain("arith.addf"));
		Assert.That(ir, Does.Contain("return %t2 : f64"));
	}

	[Test]
	public void ConditionalBranchGeneratesCmpfAndCondBr()
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
		Assert.That(ir, Does.Contain("arith.cmpf ogt"));
		Assert.That(ir, Does.Contain("cf.cond_br"));
	}

	[Test]
	public void UnconditionalJumpGeneratesCfBr()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 1.0)),
			new Jump(1),
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 99.0)),
			new ReturnInstruction(Register.R0)
		};
		var ir = compiler.CompileInstructions("UncondJump", instructions);
		Assert.That(ir, Does.Contain("cf.br ^bb"));
	}

	[Test]
	public void NotEqualComparisonUsesOnePrefix()
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
		Assert.That(ir, Does.Contain("arith.cmpf one"));
	}

	[Test]
	public void LessThanComparisonUsesOlt()
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
		Assert.That(ir, Does.Contain("arith.cmpf olt"));
	}

	[Test]
	public void EqualComparisonUsesOeq()
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
		Assert.That(ir, Does.Contain("arith.cmpf oeq"));
	}

	[Test]
	public void CompileForPlatformWrapsInModuleWithMainEntry()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new ReturnInstruction(Register.R0)
		};
		var mlir = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		Assert.That(mlir, Does.Contain("module {"));
		Assert.That(mlir, Does.Contain("func.func @Run("));
		Assert.That(mlir, Does.Contain("func.func @main() -> i32"));
		Assert.That(mlir, Does.Contain("func.call @Run()"));
		Assert.That(mlir, Does.Contain("return %zero : i32"));
	}

	[Test]
	public void MlirOutputDoesNotContainPlatformSpecificAssembly()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var mlir = compiler.CompileForPlatform("Compute", instructions, Platform.Linux);
		Assert.That(mlir, Does.Not.Contain("xmm"));
		Assert.That(mlir, Does.Not.Contain("movsd"));
		Assert.That(mlir, Does.Not.Contain("section .text"));
		Assert.That(mlir, Does.Not.Contain("target triple"));
	}

	[Test]
	public void MlirUsesArithDialectNotLlvmIr()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var mlir = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		Assert.That(mlir, Does.Contain("arith.addf"));
		Assert.That(mlir, Does.Contain("arith.constant"));
		Assert.That(mlir, Does.Not.Contain("fadd double"));
		Assert.That(mlir, Does.Not.Contain("define double"));
	}

	[Test]
	public void CompileForPlatformWithPrintUsesLlvmPrintfAndStringGlobal()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 42.0)),
			new PrintInstruction("Result: ", Register.R0),
			new ReturnInstruction(Register.R0)
		};
		var mlir = compiler.CompileForPlatform("PrintRun", instructions, Platform.Windows);
		Assert.That(mlir, Does.Contain("llvm.func @printf(!llvm.ptr, ...) -> i32"));
		Assert.That(mlir, Does.Contain("llvm.mlir.global internal constant @str_PrintRun_0"));
		Assert.That(mlir, Does.Contain("Result: %g\\0A\\00"));
		Assert.That(mlir, Does.Contain("llvm.call @printf("));
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
	public void MlirLinkerIsAvailableDoesNotThrow() =>
		Assert.DoesNotThrow(() => _ = MlirLinker.IsAvailable);

	[Test]
	public void PureAdderStyleTypeGeneratesMlirWithReturnConstant()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("MlirPureAdder",
			"has first Number",
			"has second Number",
			"Add Number",
			"\tfirst + second",
			"Run Number",
			"\t42")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var mlir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux);
		Assert.That(mlir, Does.Contain("func.func @MlirPureAdder("));
		Assert.That(mlir, Does.Contain("arith.constant 42.0 : f64"));
		Assert.That(mlir, Does.Contain("func.func @main() -> i32"));
	}

	[Test]
	public void SimpleCalculatorStyleTypeGeneratesAddAndMultiply()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("MlirCalc",
			"has first Number",
			"has second Number",
			"Add Number",
			"\tfirst + second",
			"Multiply Number",
			"\tfirst * second",
			"Run Number",
			"\tconstant calc = MlirCalc(2, 3)",
			"\tconstant added = calc.Add",
			"\tconstant multiplied = calc.Multiply",
			"\tadded + multiplied")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var multiplyMethod = type.Methods.First(method => method.Name == "Multiply");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = GenerateMethodInstructions(addMethod);
		var multiplyInstructions = GenerateMethodInstructions(multiplyMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BytecodeDeserializer.BuildMethodInstructionKey(type.Name, addMethod.Name,
				addMethod.Parameters.Count)] = addInstructions,
			[BytecodeDeserializer.BuildMethodInstructionKey(type.Name, multiplyMethod.Name,
				multiplyMethod.Parameters.Count)] = multiplyInstructions
		};
		var mlir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
		Assert.That(mlir, Does.Contain("func.func @MlirCalc_Add_0("));
		Assert.That(mlir, Does.Contain("func.func @MlirCalc_Multiply_0("));
		Assert.That(mlir, Does.Contain("func.call @MlirCalc_Add_0("));
		Assert.That(mlir, Does.Contain("func.call @MlirCalc_Multiply_0("));
		Assert.That(mlir, Does.Contain("arith.addf"));
	}

	[Test]
	public void AreaCalculatorStyleTypeWithMultiplyComputation()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("MlirArea",
			"has width Number",
			"has height Number",
			"Area Number",
			"\twidth * height",
			"Run Number",
			"\tconstant rect = MlirArea(5, 3)",
			"\trect.Area")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var areaMethod = type.Methods.First(method => method.Name == "Area");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var areaInstructions = GenerateMethodInstructions(areaMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BytecodeDeserializer.BuildMethodInstructionKey(type.Name, areaMethod.Name,
				areaMethod.Parameters.Count)] = areaInstructions
		};
		var mlir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
		Assert.That(mlir, Does.Contain("func.func @MlirArea_Area_0("));
		Assert.That(mlir, Does.Contain("arith.mulf"));
		Assert.That(mlir, Does.Contain("func.call @MlirArea_Area_0("));
	}

	[Test]
	public void TemperatureConverterStyleTypeWithArithmeticChain()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("MlirTempConv",
			"has celsius Number",
			"ToFahrenheit Number",
			"\tcelsius * 1.8 + 32",
			"Run Number",
			"\tconstant conv = MlirTempConv(100)",
			"\tconv.ToFahrenheit")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var toFMethod = type.Methods.First(method => method.Name == "ToFahrenheit");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var toFInstructions = GenerateMethodInstructions(toFMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BytecodeDeserializer.BuildMethodInstructionKey(type.Name, toFMethod.Name,
				toFMethod.Parameters.Count)] = toFInstructions
		};
		var mlir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
		Assert.That(mlir, Does.Contain("func.func @MlirTempConv_ToFahrenheit_0("));
		Assert.That(mlir, Does.Contain("arith.mulf"));
		Assert.That(mlir, Does.Contain("arith.addf"));
	}

	[Test]
	public void PixelStyleTypeWithDivideComputation()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("MlirPixel",
			"has red Number",
			"has green Number",
			"has blue Number",
			"Brighten Number",
			"\tred + green",
			"Darken Number",
			"\tred / 2",
			"Run Number",
			"\tconstant pixel = MlirPixel(100, 150, 200)",
			"\tpixel.Brighten")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var brightenMethod = type.Methods.First(method => method.Name == "Brighten");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var brightenInstructions = GenerateMethodInstructions(brightenMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BytecodeDeserializer.BuildMethodInstructionKey(type.Name, brightenMethod.Name,
				brightenMethod.Parameters.Count)] = brightenInstructions
		};
		var mlir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
		Assert.That(mlir, Does.Contain("func.func @MlirPixel_Brighten_0("));
		Assert.That(mlir, Does.Contain("arith.addf"));
		Assert.That(mlir, Does.Contain("func.call @MlirPixel_Brighten_0("));
	}

	[Test]
	public void ParameterizedMethodGeneratesParamSignature()
	{
		var type = new Type(TestPackage.Instance, new TypeLines("MlirArithFunc",
			"has dummy Number",
			"Add(first Number, second Number) Number",
			"\tfirst + second",
			"Run Number",
			"\tAdd(10, 20)")).ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = type.Methods.First(method => method.Name == Method.Run);
		var addMethod = type.Methods.First(method => method.Name == "Add");
		var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
		var addInstructions = GenerateMethodInstructions(addMethod);
		var precompiled = new Dictionary<string, List<Instruction>>
		{
			[BytecodeDeserializer.BuildMethodInstructionKey(type.Name, addMethod.Name,
				addMethod.Parameters.Count)] = addInstructions
		};
		var mlir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
		Assert.That(mlir, Does.Contain("func.func @MlirArithFunc_Add_2("));
		Assert.That(mlir, Does.Contain("%param0: f64"));
		Assert.That(mlir, Does.Contain("%param1: f64"));
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
		var mlir = compiler.CompileInstructions("Chain", instructions);
		Assert.That(mlir, Does.Contain("arith.addf %t0, %t1 : f64"));
		Assert.That(mlir, Does.Contain("arith.mulf %t3, %t2 : f64"));
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
		var mlir = compiler.CompileInstructions("TextIgnore", instructions);
		Assert.That(mlir, Does.Contain("arith.constant 0.0 : f64"));
		Assert.That(mlir, Does.Contain("return"));
	}

	[Test]
	public void MlirIsShorterThanNasmForSameProgram()
	{
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(NumberType, 5.0)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(NumberType, 3.0)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		};
		var mlir = compiler.CompileForPlatform("Run", instructions, Platform.Linux);
		var nasm = new InstructionsToAssembly().CompileForPlatform("Run", instructions, Platform.Linux);
		Assert.That(mlir.Length, Is.LessThan(nasm.Length),
			"MLIR should be more compact than NASM assembly");
	}

	[Test]
	public void MlirLinkerUsesSizeFocusedLoweringPasses()
	{
		var args = BuildMlirOptArgs("input.mlir", "output.llvm.mlir");
		Assert.That(args, Does.Contain("--canonicalize"));
		Assert.That(args, Does.Contain("--cse"));
		Assert.That(args, Does.Contain("--symbol-dce"));
		Assert.That(args, Does.Contain("--convert-arith-to-llvm"));
		Assert.That(args, Does.Contain("--convert-func-to-llvm"));
		Assert.That(args, Does.Contain("--convert-cf-to-llvm"));
	}

	[Test]
	public void MlirLinkerUsesSizeFocusedFlagsForLinuxWithoutPrintCalls()
	{
		var args = BuildMlirClangArgs("input.ll", "output", Platform.Linux, false);
		Assert.That(args, Does.Contain("-Oz"));
		if (OperatingSystem.IsWindows())
		{
			Assert.That(args, Does.Not.Contain("-nostdlib"));
			Assert.That(args, Does.Not.Contain("-Wl,-e,main"));
			return;
		}
		//ncrunch: no coverage start
		Assert.That(args, Does.Contain("-s"));
		Assert.That(args, Does.Contain("-nostdlib"));
		Assert.That(args, Does.Contain("-Wl,-e,main"));
		Assert.That(args, Does.Contain("-Wl,--gc-sections"));
		Assert.That(args, Does.Contain("-Wl,--strip-all"));
		Assert.That(args, Does.Contain("-Wl,--build-id=none"));
		Assert.That(args, Does.Contain("-fno-unwind-tables"));
		Assert.That(args, Does.Contain("-fno-asynchronous-unwind-tables"));
	} //ncrunch: no coverage end

	[Test]
	public void MlirLinkerAllowsStdLibForLinuxWithPrintCalls()
	{
		var args = BuildMlirClangArgs("input.ll", "output", Platform.Linux, true);
		Assert.That(args, Does.Contain("-Oz"));
		Assert.That(args, Does.Not.Contain("-nostdlib"));
		Assert.That(args, Does.Not.Contain("-Wl,-e,main"));
		if (OperatingSystem.IsWindows())
			return;
		//ncrunch: no coverage start
		Assert.That(args, Does.Contain("-s"));
		Assert.That(args, Does.Contain("-Wl,--gc-sections"));
		Assert.That(args, Does.Contain("-Wl,--strip-all"));
	} //ncrunch: no coverage end

	[Test]
	public void MlirLinkerWindowsPrintUsesKernel32WithoutCrt()
	{
		var args = BuildMlirClangArgs("input.ll", "output.exe", Platform.Windows, true);
		Assert.That(args, Does.Contain("-nostdlib"));
		Assert.That(args, Does.Contain("-lkernel32"));
		Assert.That(args, Does.Contain("-Wl,/ENTRY:main"));
		Assert.That(args, Does.Not.Contain("/SUBSYSTEM:CONSOLE"));
		Assert.That(args, Does.Not.Contain("@printf"));
	}

	[Test]
	public void MlirLinkerRewriteWindowsPrintRuntimeRemovesPrintf()
	{
		const string LlvmIr = """
			@str_PrintRun_0 = internal constant [12 x i8] c"Result: %g\0A\00"

			declare i32 @printf(ptr, ...)

			define double @PrintRun() {
			  %1 = call i32 (ptr, ...) @printf(ptr @str_PrintRun_0, double 4.200000e+01)
			  ret double 4.200000e+01
			}

			""";
		var rewritten = RewriteWindowsPrintRuntime(LlvmIr);
		Assert.That(rewritten, Does.Contain("@_fltused = global i32 0"));
		Assert.That(rewritten, Does.Contain("declare ptr @GetStdHandle(i32)"));
		Assert.That(rewritten, Does.Contain("declare i32 @WriteFile(ptr, ptr, i32, ptr, ptr)"));
		Assert.That(rewritten, Does.Contain("define void @print_number_from_double("));
		Assert.That(rewritten, Does.Not.Contain("declare i32 @printf(ptr, ...)"));
		Assert.That(rewritten, Does.Not.Contain("@printf("));
		Assert.That(rewritten, Does.Contain("@GetStdHandle(i32 -11)"));
		Assert.That(rewritten, Does.Contain("@WriteFile("));
		Assert.That(rewritten, Does.Contain("@print_number_from_double("));
	}

	private static string BuildMlirOptArgs(string inputPath, string outputPath)
	{
		var buildMethod = typeof(MlirLinker).GetMethod("BuildMlirOptArgs",
			BindingFlags.Static | BindingFlags.NonPublic);
		Assert.That(buildMethod, Is.Not.Null,
			"MlirLinker should expose a private BuildMlirOptArgs helper for consistent pass testing");
		var result = buildMethod!.Invoke(null, [inputPath, outputPath]);
		return result as string ?? throw new InvalidOperationException("Expected mlir-opt args string");
	}

	private static string BuildMlirClangArgs(string inputPath, string outputPath, Platform platform,
		bool hasPrintCalls)
	{
		var buildMethod = typeof(MlirLinker).GetMethod("BuildClangArgs",
			BindingFlags.Static | BindingFlags.NonPublic);
		Assert.That(buildMethod, Is.Not.Null,
			"MlirLinker should expose a private BuildClangArgs helper for consistent flag testing");
		var result = buildMethod!.Invoke(null, [inputPath, outputPath, platform, hasPrintCalls]);
		return result as string ?? throw new InvalidOperationException("Expected clang args string");
	}

	private static List<Instruction> GenerateMethodInstructions(Method method)
	{
		var body = method.GetBodyAndParseIfNeeded();
		var expressions = body is Body b
			? b.Expressions
			: [body];
		return new BytecodeGenerator(new InvokedMethod(expressions,
			new Dictionary<string, ValueInstance>(), method.ReturnType), new Registry()).Generate();
	}

	[Test]
	public void RangeLoopEmitsScfFor()
	{
		var startRegister = Register.R0;
		var endRegister = Register.R1;
		var loopBegin = new LoopBeginInstruction(startRegister, endRegister);
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(startRegister, new ValueInstance(NumberType, 0.0)),
			new LoadConstantInstruction(endRegister, new ValueInstance(NumberType, 10.0)),
			loopBegin,
			new LoadConstantInstruction(Register.R2, new ValueInstance(NumberType, 1.0)),
			new LoopEndInstruction(2) { Begin = loopBegin },
			new ReturnInstruction(Register.R2)
		};
		var mlir = compiler.CompileInstructions("LoopTest", instructions);
		Assert.That(mlir, Does.Contain("scf.for"));
		Assert.That(mlir, Does.Contain("arith.constant 0.0"));
		Assert.That(mlir, Does.Contain("arith.constant 10.0"));
	}

	[Test]
	public void ParallelHintEmitsScfParallel()
	{
		var startRegister = Register.R0;
		var endRegister = Register.R1;
		var loopBegin = new LoopBeginInstruction(startRegister, endRegister);
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(startRegister, new ValueInstance(NumberType, 0.0)),
			new LoadConstantInstruction(endRegister, new ValueInstance(NumberType, 8294400.0)),
			loopBegin,
			new LoadConstantInstruction(Register.R2, new ValueInstance(NumberType, 1.0)),
			new LoopEndInstruction(2) { Begin = loopBegin },
			new ReturnInstruction(Register.R2)
		};
		var mlir = compiler.CompileInstructions("ParallelLoopTest", instructions);
		Assert.That(mlir, Does.Contain("scf.parallel"),
			"Large loops (>100K iterations) should emit scf.parallel for automatic CPU/GPU parallelism");
	}

	[Test]
	public void SmallLoopDoesNotParallelize()
	{
		var startRegister = Register.R0;
		var endRegister = Register.R1;
		var loopBegin = new LoopBeginInstruction(startRegister, endRegister);
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(startRegister, new ValueInstance(NumberType, 0.0)),
			new LoadConstantInstruction(endRegister, new ValueInstance(NumberType, 100.0)),
			loopBegin,
			new LoadConstantInstruction(Register.R2, new ValueInstance(NumberType, 1.0)),
			new LoopEndInstruction(2) { Begin = loopBegin },
			new ReturnInstruction(Register.R2)
		};
		var mlir = compiler.CompileInstructions("SmallLoopTest", instructions);
		Assert.That(mlir, Does.Contain("scf.for"));
		Assert.That(mlir, Does.Not.Contain("scf.parallel"),
			"Small loops (<100K iterations) should NOT parallelize");
	}

	[Test]
	public void MlirOptArgsIncludeScfToCfConversion()
	{
		var args = BuildMlirOptArgs("test.mlir", "test.llvm.mlir");
		Assert.That(args, Does.Contain("--convert-scf-to-cf"),
			"MLIR opt should convert SCF for-loops to control flow");
	}

	[Test]
	public void ComplexBodyWithFewerIterationsStillParallelizes()
	{
		var startRegister = Register.R0;
		var endRegister = Register.R1;
		var loopBegin = new LoopBeginInstruction(startRegister, endRegister);
		var bodyInstructions = new List<Instruction>();
		for (var bodyIndex = 0; bodyIndex < 20; bodyIndex++)
			bodyInstructions.Add(new BinaryInstruction(InstructionType.Add, Register.R2, Register.R3,
				Register.R4));
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(startRegister, new ValueInstance(NumberType, 0.0)),
			new LoadConstantInstruction(endRegister, new ValueInstance(NumberType, 10000.0)),
			new LoadConstantInstruction(Register.R2, new ValueInstance(NumberType, 1.0)),
			new LoadConstantInstruction(Register.R3, new ValueInstance(NumberType, 2.0)),
			loopBegin
		};
		instructions.AddRange(bodyInstructions);
		instructions.Add(new LoopEndInstruction(bodyInstructions.Count) { Begin = loopBegin });
		instructions.Add(new ReturnInstruction(Register.R2));
		var mlir = compiler.CompileInstructions("ComplexBodyTest", instructions);
		Assert.That(mlir, Does.Contain("scf.parallel"),
			$"10K iterations × 20 body instructions = 200K complexity > {InstructionsToMlir.ComplexityThreshold} threshold → parallel");
	}

	[Test]
	public void SimpleBodyWithManyIterationsDoesNotParallelizeIfComplexityBelowThreshold()
	{
		var startRegister = Register.R0;
		var endRegister = Register.R1;
		var loopBegin = new LoopBeginInstruction(startRegister, endRegister);
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(startRegister, new ValueInstance(NumberType, 0.0)),
			new LoadConstantInstruction(endRegister, new ValueInstance(NumberType, 50000.0)),
			loopBegin,
			new LoadConstantInstruction(Register.R2, new ValueInstance(NumberType, 1.0)),
			new LoopEndInstruction(1) { Begin = loopBegin },
			new ReturnInstruction(Register.R2)
		};
		var mlir = compiler.CompileInstructions("SimpleBodyTest", instructions);
		Assert.That(mlir, Does.Contain("scf.for"));
		Assert.That(mlir, Does.Not.Contain("scf.parallel"),
			$"50K iterations × 1 body instruction = 50K complexity < {InstructionsToMlir.ComplexityThreshold} threshold → sequential");
	}

	private static string RewriteWindowsPrintRuntime(string llvmIr)
	{
		var rewriteMethod = typeof(MlirLinker).GetMethod("RewriteWindowsPrintRuntime",
			BindingFlags.Static | BindingFlags.NonPublic);
		Assert.That(rewriteMethod, Is.Not.Null,
			"MlirLinker should expose a private RewriteWindowsPrintRuntime helper for Windows no-CRT print rewriting");
		var result = rewriteMethod!.Invoke(null, [llvmIr]);
		return result as string ?? throw new InvalidOperationException("Expected rewritten LLVM IR string");
	}
}
