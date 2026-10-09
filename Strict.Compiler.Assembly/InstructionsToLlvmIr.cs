using System.Globalization;
using System.Text;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Language;

namespace Strict.Compiler.Assembly;

/// <summary>
/// Compiles Strict bytecode instructions to LLVM IR text. Generates typed SSA form IR that can be
/// compiled directly by clang or llc, benefiting from LLVM's optimization passes (-O2 by default)
/// and platform-specific code generation. Much simpler than raw NASM: no manual register allocation,
/// no ABI handling, no stack frame management — LLVM handles all of this.
/// </summary>
public sealed partial class InstructionsToLlvmIr : InstructionsCompiler
{
	public override Task<string> Compile(BinaryExecutable binary, Platform platform)
	{
		var precompiledMethods = BuildPrecompiledMethodsInternal(binary);
		var output = CompileForPlatform(Method.Run, binary.EntryPoint.instructions, platform,
			precompiledMethods, binary);
		return Task.FromResult(output);
	}

	public override string Extension => ".ll";

	public string CompileInstructions(string methodName, List<Instruction> instructions) =>
		BuildFunction(methodName, [], instructions, Platform.Linux);

	private static string CompileForPlatform(string methodName,
		IReadOnlyList<Instruction> instructions, Platform platform,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods = null,
		BinaryExecutable? binary = null)
	{
		var hasPrint = instructions.OfType<PrintInstruction>().Any();
		var methodInfos = CollectMethods([.. instructions], precompiledMethods, binary);
		var hasNumericPrint = HasNumericPrint(instructions) ||
			methodInfos.Values.Any(info => HasNumericPrint(info.Instructions));
		var module = BuildModuleHeader(platform, hasPrint, hasNumericPrint);
		module += "\n" + BuildFunction(methodName, [], [.. instructions], platform, methodInfos);
		foreach (var methodInfo in methodInfos.Values)
			module += "\n" + BuildFunction(methodInfo.Symbol, methodInfo.ParameterNames,
				methodInfo.Instructions, platform, methodInfos);
		module += "\n" + BuildEntryPoint(methodName);
		if (platform == Platform.Windows && hasNumericPrint)
			module += "\n" + BuildWindowsPrintNumberHelper();
		var stringConstants = CollectPrintStrings([.. instructions], platform);
		foreach (var methodInfo in methodInfos.Values)
		foreach (var (label, text) in CollectPrintStrings(methodInfo.Instructions, platform))
			//ncrunch: no coverage start
			if (stringConstants.All(existing => existing.Label != label))
				stringConstants.Add((label, text));
		//ncrunch: no coverage end
		if (stringConstants.Count > 0)
			module += "\n" + BuildStringConstants(stringConstants);
		return module;
	}

	private static string BuildModuleHeader(Platform platform, bool hasPrint, bool hasNumericPrint)
	{
		var targetTriple = platform switch
		{
			Platform.Windows => "x86_64-pc-windows-msvc",
			Platform.Linux => "x86_64-unknown-linux-gnu",
			Platform.MacOS => "x86_64-apple-macosx",
			_ => throw new UnsupportedPlatform(platform) //ncrunch: no coverage
		};
		var header = $"target triple = \"{targetTriple}\"\n";
		if (platform == Platform.Windows)
			header += "@_fltused = global i32 0\n";
		if (hasPrint)
			if (platform == Platform.Windows)
			{
				header += "\ndeclare ptr @GetStdHandle(i32)\n";
				header += "declare i32 @WriteFile(ptr, ptr, i32, ptr, ptr)\n";
			}
			else
			{
				header += "\ndeclare i32 @printf(ptr, ...)\n";
				if (hasNumericPrint)
				{
					header += "declare i32 @snprintf(ptr, i64, ptr, ...)\n";
					header += "@str.safe_s = private unnamed_addr constant [3 x i8] c\"%s\\00\"\n";
				}
			}
		return header;
	}

	private static string BuildFunction(string methodName, IEnumerable<string> paramNames,
		List<Instruction> instructions, Platform platform,
		Dictionary<string, CompiledMethodInfo>? compiledMethods = null)
	{
		var parameterList = paramNames.ToList();
		var paramIndexByName = parameterList.Select((name, index) => (name, index)).
			ToDictionary(x => x.name, x => x.index);
		var paramSignature =
			string.Join(", ", parameterList.Select((_, index) => $"double %param{index}"));
		var lines = new List<string> { $"define double @{methodName}({paramSignature}) {{", "entry:" };
		var context = new EmitContext(paramIndexByName, instructions, compiledMethods, platform);
		for (var index = 0; index < instructions.Count; index++)
		{
			if (context.BlockLabels.TryGetValue(index, out var label))
			{
				if (!context.TerminatedBlocks.Contains(context.CurrentBlock))
					lines.Add($"  br label %{label}");
				lines.Add($"{label}:");
				context.CurrentBlock = label;
			}
			EmitInstruction(instructions[index], lines, context, index);
		}
		if (!context.HasReturn)
			lines.Add("  ret double 0.0"); //ncrunch: no coverage
		lines.Add("}");
		return string.Join("\n", lines);
	}

	private sealed class EmitContext(Dictionary<string, int> paramIndexByName,
		List<Instruction> instructions,
		Dictionary<string, CompiledMethodInfo>? compiledMethods,
		Platform platform)
	{
		public Dictionary<string, int> ParamIndexByName { get; } = paramIndexByName;
		public Dictionary<string, CompiledMethodInfo>? CompiledMethods { get; } = compiledMethods;
		public Platform Platform { get; } = platform;
		public Dictionary<Register, Register[]> RegisterInstances { get; } = new();
		public Dictionary<string, Register[]> VariableInstances { get; } = new(StringComparer.Ordinal);
		public Dictionary<int, string> BlockLabels { get; } = BuildBlockLabels(instructions);
		public Dictionary<int, int> JumpEndPositions { get; } = BuildJumpEndPositions(instructions);
		public Dictionary<Register, string> RegisterValues { get; } = new();
		public Dictionary<string, string> VariablePointers { get; } = new(StringComparer.Ordinal);
		public HashSet<string> AllocatedVariables { get; } = new(StringComparer.Ordinal);
		public HashSet<string> TerminatedBlocks { get; } = new(StringComparer.Ordinal);
		public string CurrentBlock { get; set; } = "entry";
		public string? LastConditionTemp { get; set; }
		public int TempCounter { get; set; }
		public bool HasReturn { get; set; }
		public string NextTemp() => $"%t{TempCounter++}";
	}

	private static void EmitInstruction(Instruction instruction, List<string> lines,
		EmitContext context, int index)
	{
		switch (instruction.InstructionType)
		{
		case InstructionType.StoreConstantToVariable:
			var storeConst = (StoreVariableInstruction)instruction;
			if (!context.ParamIndexByName.ContainsKey(storeConst.Identifier))
				EmitStoreVariable(storeConst, lines, context);
			break; //ncrunch: no coverage
		case InstructionType.LoadVariableToRegister:
			var loadVar = (LoadVariableToRegister)instruction;
			EmitLoadVariable(loadVar, lines, context);
			break;
		case InstructionType.LoadConstantToRegister:
			var loadConst = (LoadConstantInstruction)instruction;
			EmitLoadConstant(loadConst, context);
			break;
		case InstructionType.Add:
		case InstructionType.Subtract:
		case InstructionType.Multiply:
		case InstructionType.Divide:
		case InstructionType.Modulo:
			EmitArithmetic((BinaryInstruction)instruction, lines, context);
			break;
		case InstructionType.Equal:
		case InstructionType.NotEqual:
		case InstructionType.LessThan:
		case InstructionType.GreaterThan:
			EmitComparison((BinaryInstruction)instruction, lines, context);
			break;
		case InstructionType.StoreRegisterToVariable:
			var storeReg = (StoreFromRegisterInstruction)instruction;
			EmitStoreFromRegister(storeReg, lines, context);
			break;
		case InstructionType.Return:
			var ret = (ReturnInstruction)instruction;
			EmitReturn(ret, lines, context);
			break;
		case InstructionType.Print:
			var print = (PrintInstruction)instruction;
			EmitPrint(print, lines, context);
			break;
		case InstructionType.Jump:
		case InstructionType.JumpIfTrue:
		case InstructionType.JumpIfFalse:
			var jump = (Jump)instruction;
			EmitJump(jump, lines, context, index);
			break;
		case InstructionType.Invoke:
			var invoke = (Invoke)instruction;
			EmitInvoke(invoke, lines, context);
			break;
		case InstructionType.JumpEnd:
			break; //ncrunch: no coverage
		case InstructionType.JumpToIdIfFalse:
		case InstructionType.JumpToIdIfTrue:
			var jumpToId = (JumpToId)instruction;
			EmitJumpToId(jumpToId, lines, context, index); //ncrunch: no coverage
			break; //ncrunch: no coverage
		default:
			throw new NotSupportedByBackend($"LLVM IR compilation does not support instruction: {
				instruction.GetType().Name
			} ({
				instruction.InstructionType
			})");
		}
	}

	private static void EmitStoreVariable(StoreVariableInstruction store, List<string> lines,
		EmitContext context)
	{
		if (store.ValueInstance.IsText)
			return;
		//ncrunch: no coverage start
		EnsureVariableAllocated(store.Identifier, lines, context);
		var value = FormatDouble(store.ValueInstance.Number);
		lines.Add($"  store double {value}, ptr {context.VariablePointers[store.Identifier]}");
	} //ncrunch: no coverage end

	private static void EnsureVariableAllocated(string name, List<string> lines, EmitContext context)
	{
		if (context.AllocatedVariables.Add(name))
		{
			var pointer = $"%var.{name}";
			context.VariablePointers[name] = pointer;
			lines.Insert(FindEntryInsertPoint(lines), $"  {pointer} = alloca double");
		}
	}

	private static int FindEntryInsertPoint(List<string> lines)
	{
		for (var index = 0; index < lines.Count; index++)
			if (lines[index] == "entry:")
				return index + 1;
		return 1; //ncrunch: no coverage
	}

	private static void EmitLoadVariable(LoadVariableToRegister loadVar, List<string> lines,
		EmitContext context)
	{
		if (context.VariableInstances.TryGetValue(loadVar.Identifier, out var instances))
		{ //ncrunch: no coverage start
			context.RegisterInstances[loadVar.Register] = instances;
			return;
		} //ncrunch: no coverage end
		if (context.ParamIndexByName.TryGetValue(loadVar.Identifier, out var paramIndex))
		{
			context.RegisterValues[loadVar.Register] = $"%param{paramIndex}";
			return;
		}
		if (context.VariablePointers.TryGetValue(loadVar.Identifier, out var pointer))
		{
			var temp = context.NextTemp();
			lines.Add($"  {temp} = load double, ptr {pointer}");
			context.RegisterValues[loadVar.Register] = temp;
		}
	}

	private static void EmitLoadConstant(LoadConstantInstruction loadConst, EmitContext context)
	{
		if (!loadConst.Constant.IsText)
			context.RegisterValues[loadConst.Register] = FormatDouble(loadConst.Constant.Number);
	}

	private static void EmitArithmetic(BinaryInstruction binary, List<string> lines,
		EmitContext context)
	{
		var left = GetRegisterValue(binary.Registers[0], context);
		var right = GetRegisterValue(binary.Registers[1], context);
		var dest = context.NextTemp();
		var op = binary.InstructionType switch
		{
			InstructionType.Add => "fadd",
			InstructionType.Subtract => "fsub",
			InstructionType.Multiply => "fmul",
			InstructionType.Divide => "fdiv",
			InstructionType.Modulo => "frem",
			_ => throw new NotSupportedByBackend( //ncrunch: no coverage
				$"LLVM IR compilation of {binary.InstructionType} is not supported")
		};
		lines.Add($"  {dest} = {op} double {left}, {right}");
		context.RegisterValues[binary.Registers[^1]] = dest;
	}

	private static void EmitComparison(BinaryInstruction binary, List<string> lines,
		EmitContext context)
	{
		var left = GetRegisterValue(binary.Registers[0], context);
		var right = GetRegisterValue(binary.Registers[1], context);
		var predicate = binary.InstructionType switch
		{
			InstructionType.Equal => "oeq",
			InstructionType.NotEqual => "one",
			InstructionType.LessThan => "olt",
			InstructionType.GreaterThan => "ogt",
			_ => throw new NotSupportedByBackend( //ncrunch: no coverage
				$"LLVM IR comparison {binary.InstructionType} is not supported")
		};
		var temp = context.NextTemp();
		lines.Add($"  {temp} = fcmp {predicate} double {left}, {right}");
		context.LastConditionTemp = temp;
	}

	private static void EmitStoreFromRegister(StoreFromRegisterInstruction storeReg,
		List<string> lines, EmitContext context)
	{
		if (context.RegisterInstances.TryGetValue(storeReg.Register, out var constructorArgs))
		{
			context.VariableInstances[storeReg.Identifier] = constructorArgs;
			return;
		}
		EnsureVariableAllocated(storeReg.Identifier, lines, context);
		var value = GetRegisterValue(storeReg.Register, context);
		lines.Add($"  store double {value}, ptr {context.VariablePointers[storeReg.Identifier]}");
	}

	private static string BuildEntryPoint(string methodName) =>
		string.Join("\n",
			new[]
			{
				"define i32 @main() {", "entry:", $"  %result = call double @{methodName}()",
				"  ret i32 0", "}"
			});

	private static string GetRegisterValue(Register register, EmitContext context) =>
		context.RegisterValues.GetValueOrDefault(register, "0.0");

	private static string FormatDouble(double value) =>
		value == 0.0
			? "0.0"
			: value == (long)value
				? $"{value:F1}"
				: value.ToString("G17", CultureInfo.InvariantCulture);
}
