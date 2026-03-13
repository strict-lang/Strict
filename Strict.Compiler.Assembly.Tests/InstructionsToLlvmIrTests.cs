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
var type = new Type(TestPackage.Instance, new TypeLines("LlvmInvokeType",
"has dummy Number",
"Add(first Number, second Number) Number",
"\tfirst + second",
"Run Number",
"\tAdd(2, 3)")).ParseMembersAndMethods(new MethodExpressionParser());
var runMethod = type.Methods.First(method => method.Name == Method.Run);
var addMethod = type.Methods.First(method => method.Name == "Add");
var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
var addInstructions = GenerateMethodInstructions(addMethod);
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
public void PrintInstructionDeclaresprintfAndUsesGep()
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
var addInstructions = GenerateMethodInstructions(addMethod);
var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(type.Name, addMethod.Name,
addMethod.Parameters.Count);
var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux,
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
var type = new Type(TestPackage.Instance, new TypeLines("LlvmPureAdder",
"has first Number",
"has second Number",
"Add Number",
"\tfirst + second",
"Run Number",
"\t42")).ParseMembersAndMethods(new MethodExpressionParser());
var runMethod = type.Methods.First(method => method.Name == Method.Run);
var runInstructions = new BytecodeGenerator(new MethodCall(runMethod)).Generate();
var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux);
Assert.That(ir, Does.Contain("define double @LlvmPureAdder("));
Assert.That(ir, Does.Contain("ret double 42.0"));
Assert.That(ir, Does.Contain("define i32 @main()"));
}

[Test]
public void SimpleCalculatorStyleTypeGeneratesAddAndMultiplyFunctions()
{
var type = new Type(TestPackage.Instance, new TypeLines("LlvmCalc",
"has first Number",
"has second Number",
"Add Number",
"\tfirst + second",
"Multiply Number",
"\tfirst * second",
"Run Number",
"\tconstant calc = LlvmCalc(2, 3)",
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
var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Windows, precompiled);
Assert.That(ir, Does.Contain("define double @LlvmCalc_Add_0("));
Assert.That(ir, Does.Contain("define double @LlvmCalc_Multiply_0("));
Assert.That(ir, Does.Contain("call double @LlvmCalc_Add_0("));
Assert.That(ir, Does.Contain("call double @LlvmCalc_Multiply_0("));
Assert.That(ir, Does.Contain("fadd double"));
}

[Test]
public void ArithmeticFunctionStyleTypeWithParametersGeneratesParamSignature()
{
var type = new Type(TestPackage.Instance, new TypeLines("LlvmArithFunc",
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
var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
Assert.That(ir, Does.Contain("define double @LlvmArithFunc_Add_2("));
Assert.That(ir, Does.Contain("double %param0"));
Assert.That(ir, Does.Contain("double %param1"));
}

[Test]
public void AreaCalculatorStyleTypeWithMultiplyComputation()
{
var type = new Type(TestPackage.Instance, new TypeLines("LlvmArea",
"has width Number",
"has height Number",
"Area Number",
"\twidth * height",
"Run Number",
"\tconstant rect = LlvmArea(5, 3)",
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
var ir = compiler.CompileForPlatform(type.Name, runInstructions, Platform.Linux, precompiled);
Assert.That(ir, Does.Contain("define double @LlvmArea_Area_0("));
Assert.That(ir, Does.Contain("fmul double"));
Assert.That(ir, Does.Contain("call double @LlvmArea_Area_0("));
}

[Test]
public void TemperatureConverterStyleTypeWithArithmeticChain()
{
var type = new Type(TestPackage.Instance, new TypeLines("LlvmTempConv",
"has celsius Number",
"ToFahrenheit Number",
"\tcelsius * 1.8 + 32",
"Run Number",
"\tconstant conv = LlvmTempConv(100)",
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
public void ToolRunnerFindToolReturnsNullForMissingTool() =>
Assert.That(ToolRunner.FindTool("nonexistent_tool_xyz"), Is.Null);

[Test]
public void ToolRunnerEnsureOutputFileExistsThrowsForMissingFile()
{
var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".missing");
Assert.Throws<InvalidOperationException>(() =>
ToolRunner.EnsureOutputFileExists(path, "test", Platform.Linux));
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
}
