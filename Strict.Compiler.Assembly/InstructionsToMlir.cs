using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Language;

namespace Strict.Compiler.Assembly;

/// <summary>
/// Compiles Strict bytecode instructions to MLIR text using the arith and func dialects.
/// MLIR is a higher-level IR than LLVM IR — arith.addf/mulf/divf map directly to Strict operations,
/// and MLIR's pass pipeline handles lowering to LLVM dialect then to LLVM IR automatically.
/// Pipeline: bytecode → .mlir text → mlir-opt (lower to LLVM) → mlir-translate (to .ll) → clang → executable
/// </summary>
public sealed class InstructionsToMlir : InstructionsCompiler
{
	public override Task<string> Compile(BinaryExecutable binary, Platform platform)
	{
		var precompiledMethods = BuildPrecompiledMethodsInternal(binary);
		var output = CompileForPlatform(Method.Run, binary.EntryPoint.instructions,
			precompiledMethods, binary);
		return Task.FromResult(output);
	}

	public override string Extension => ".mlir";

	public string CompileInstructions(string methodName, List<Instruction> instructions) =>
		BuildFunction(methodName, [], instructions).Text;

	private static string CompileForPlatform(string methodName, List<Instruction> instructions,
		Dictionary<string, List<Instruction>>? precompiledMethods = null,
		BinaryExecutable? binary = null)
	{
		var hasPrint = instructions.OfType<PrintInstruction>().Any();
		var methodInfos = CollectMethods([.. instructions], precompiledMethods, binary);
		var allStringConstants = new List<(string Name, string Text, int ByteLen)>();
		var entryFunction = BuildFunction(methodName, [], [.. instructions], methodInfos);
		allStringConstants.AddRange(entryFunction.StringConstants);
		var hasGpuOps = entryFunction.UsesGpu;
		var methodFunctions = new List<CompiledFunction>();
		foreach (var methodInfo in methodInfos.Values)
		{
			var methodFunction = BuildFunction(methodInfo.Symbol, methodInfo.ParameterNames,
				methodInfo.Instructions, methodInfos);
			allStringConstants.AddRange(methodFunction.StringConstants);
			hasGpuOps |= methodFunction.UsesGpu;
			methodFunctions.Add(methodFunction);
		}
		var module = hasGpuOps
			? "module attributes {gpu.container_module} {\n"
			: "module {\n";
		if (hasPrint ||
			methodInfos.Values.Any(info => info.Instructions.OfType<PrintInstruction>().Any()))
			module += BuildPrintfDeclarations(); //ncrunch: no coverage
		module += entryFunction.Text;
		foreach (var methodFunction in methodFunctions)
			module += "\n" + methodFunction.Text;
		if (allStringConstants.Count > 0)
			module += "\n" + BuildStringGlobals(allStringConstants);
		module += "\n" + BuildEntryPoint(methodName);
		module += "\n}\n";
		return module;
	}

	private readonly record struct CompiledFunction(string Text,
		List<(string Name, string Text, int ByteLen)> StringConstants, bool UsesGpu = false);

	private static string BuildPrintfDeclarations() =>
		"  llvm.func @printf(!llvm.ptr, ...) -> i32\n";

	private static string BuildStringGlobals(
		IReadOnlyList<(string Name, string Text, int ByteLen)> stringConstants) =>
		string.Join("\n", stringConstants.Select(stringConstant =>
			$"  llvm.mlir.global internal constant {stringConstant.Name}(\"{stringConstant.Text}\") : !llvm.array<{stringConstant.ByteLen} x i8>"));

	private static CompiledFunction BuildFunction(string methodName, IEnumerable<string> paramNames,
		List<Instruction> instructions, Dictionary<string, CompiledMethodInfo>? compiledMethods = null)
	{
		var context = new EmitContext(methodName);
		var paramList = paramNames.ToList();
		for (var index = 0; index < paramList.Count; index++)
			context.ParamIndexByName[paramList[index]] = index;
		var paramSignature = paramList.Count > 0
			? string.Join(", ", paramList.Select((_, index) => $"%param{index}: f64"))
			: "";
		var lines = new List<string> { $"  func.func @{methodName}({paramSignature}) -> f64 {{" };
		for (var index = 0; index < instructions.Count; index++)
			EmitInstruction(instructions, index, lines, context, compiledMethods);
		if (!instructions.Any(instr => instr is ReturnInstruction))
		{ //ncrunch: no coverage start
			lines.Add("    %zero = arith.constant 0.0 : f64");
			lines.Add("    return %zero : f64");
		} //ncrunch: no coverage end
		lines.Add("  }");
		return new CompiledFunction(string.Join("\n", lines), context.StringConstants, context.HadGpuOps);
	}

	private static void EmitInstruction(List<Instruction> instructions, int index,
		List<string> lines, EmitContext context,
		Dictionary<string, CompiledMethodInfo>? compiledMethods)
	{
		var instruction = instructions[index];
		if (context.JumpTargets.Contains(index))
			lines.Add($"  ^bb{index}:");
		switch (instruction.InstructionType)
		{
		case InstructionType.LoadConstantToRegister:
			var loadConst = (LoadConstantInstruction)instruction;
			EmitLoadConstant(loadConst, lines, context);
			break;
		case InstructionType.Add:
		case InstructionType.Subtract:
		case InstructionType.Multiply:
		case InstructionType.Divide:
		case InstructionType.Modulo:
		case InstructionType.Equal:
		case InstructionType.NotEqual:
		case InstructionType.LessThan:
		case InstructionType.GreaterThan:
			var binary = (BinaryInstruction)instruction;
			EmitBinary(binary, lines, context);
			break;
		case InstructionType.Return:
			var ret = (ReturnInstruction)instruction;
			EmitReturn(ret, lines, context);
			break;
		case InstructionType.StoreRegisterToVariable:
			var storeReg = (StoreFromRegisterInstruction)instruction;
			EmitStoreFromRegister(storeReg, context);
			break;
		case InstructionType.LoadVariableToRegister:
			var loadVar = (LoadVariableToRegister)instruction;
			EmitLoadVariable(loadVar, context);
			break;
		case InstructionType.StoreConstantToVariable:
			var storeVar = (StoreVariableInstruction)instruction;
			EmitStoreVariable(storeVar, context);
			break;
		case InstructionType.Jump:
		case InstructionType.JumpIfTrue:
		case InstructionType.JumpIfFalse:
			var jump = (Jump)instruction;
			EmitJump(jump, lines, context, index);
			break;
		case InstructionType.Print:
			var print = (PrintInstruction)instruction;
			EmitPrint(print, lines, context); //ncrunch: no coverage
			break; //ncrunch: no coverage
		case InstructionType.Invoke:
			var invoke = (Invoke)instruction;
			EmitInvoke(invoke, lines, context, compiledMethods);
			break;
		case InstructionType.JumpEnd:
		case InstructionType.JumpToIdIfFalse:
		case InstructionType.JumpToIdIfTrue:
			var jumpToId = (JumpToId)instruction;
			EmitJumpToId(jumpToId, lines, context, index); //ncrunch: no coverage
			break; //ncrunch: no coverage
		case InstructionType.LoopBegin:
			var loopBegin = (LoopBeginInstruction)instruction;
			EmitLoopBegin(loopBegin, lines, context, instructions, index);
			break;
		case InstructionType.LoopEnd:
			EmitLoopEnd(lines, context);
			break;
		default:
			throw new NotSupportedException($"MLIR compilation does not support instruction: {
				instruction.GetType().Name
			} ({
				instruction.InstructionType
			})");
		}
	}

	private static void EmitLoadConstant(LoadConstantInstruction loadConst, List<string> lines,
		EmitContext context)
	{
		if (loadConst.Constant.IsText)
			return; //ncrunch: no coverage
		var value = FormatDouble(loadConst.Constant.Number);
		var temp = context.NextTemp();
		lines.Add($"    {temp} = arith.constant {value} : f64");
		context.RegisterValues[loadConst.Register] = temp;
		context.RegisterConstants[loadConst.Register] = loadConst.Constant.Number;
	}

	private static void EmitBinary(BinaryInstruction binary, List<string> lines,
		EmitContext context)
	{
		var left = context.RegisterValues.GetValueOrDefault(binary.Registers[0], "%zero");
		var right = context.RegisterValues.GetValueOrDefault(binary.Registers[1], "%zero");
		if (IsComparison(binary.InstructionType))
		{
			EmitComparison(binary, lines, context, left, right);
			return;
		}
		var temp = context.NextTemp();
		var op = binary.InstructionType switch
		{
			InstructionType.Add => "arith.addf",
			InstructionType.Subtract => "arith.subf",
			InstructionType.Multiply => "arith.mulf",
			InstructionType.Divide => "arith.divf",
			InstructionType.Modulo => "arith.remf",
			_ => throw new NotSupportedException( //ncrunch: no coverage
				"Unsupported binary op: " + binary.InstructionType)
		};
		lines.Add($"    {temp} = {op} {left}, {right} : f64");
		if (binary.Registers.Length > 2)
			context.RegisterValues[binary.Registers[^1]] = temp;
		else
			context.RegisterValues[binary.Registers[0]] = temp; //ncrunch: no coverage
	}

	private static bool IsComparison(InstructionType type) =>
		type is InstructionType.GreaterThan or InstructionType.LessThan
			or InstructionType.Equal or InstructionType.NotEqual;

	private static void EmitComparison(BinaryInstruction binary, List<string> lines,
		EmitContext context, string left, string right)
	{
		var predicate = binary.InstructionType switch
		{
			InstructionType.GreaterThan => "ogt",
			InstructionType.LessThan => "olt",
			InstructionType.Equal => "oeq",
			InstructionType.NotEqual => "one",
			_ => "oeq" //ncrunch: no coverage
		};
		var temp = context.NextTemp();
		lines.Add($"    {temp} = arith.cmpf {predicate}, {left}, {right} : f64");
		context.LastConditionTemp = temp;
	}

	private static void EmitReturn(ReturnInstruction ret, List<string> lines, EmitContext context)
	{
		var value = context.RegisterValues.GetValueOrDefault(ret.Register, "0.0");
		if (value.StartsWith('%'))
			lines.Add($"    return {value} : f64");
		else
		{ //ncrunch: no coverage start
			var temp = $"%ret_{context.TempCounter++}";
			lines.Add($"    {temp} = arith.constant {value} : f64");
			lines.Add($"    return {temp} : f64");
		} //ncrunch: no coverage end
	}

	private static void EmitStoreFromRegister(StoreFromRegisterInstruction storeReg,
		EmitContext context)
	{
		if (context.RegisterValues.TryGetValue(storeReg.Register, out var value))
			context.VariableValues[storeReg.Identifier] = value;
		//unused: if (context.RegisterInstances.TryGetValue(storeReg.Register, out var instances))
		//unused:	context.VariableInstances[storeReg.Identifier] = instances;
	}

	private static void EmitLoadVariable(LoadVariableToRegister loadVar, EmitContext context)
	{
		if (context.VariableValues.TryGetValue(loadVar.Identifier, out var value))
			context.RegisterValues[loadVar.Register] = value;
		else if (context.ParamIndexByName.TryGetValue(loadVar.Identifier, out var paramIndex))
			context.RegisterValues[loadVar.Register] = $"%param{paramIndex}";
	}

	private static void EmitStoreVariable(StoreVariableInstruction storeVar, EmitContext context)
	{
		if (!storeVar.ValueInstance.IsText)
			context.VariableValues[storeVar.Identifier] = //ncrunch: no coverage
				FormatDouble(storeVar.ValueInstance.Number);
	}

	private static void EmitJump(Jump jump, List<string> lines, EmitContext context, int currentIndex)
	{
		var targetIndex = currentIndex + 1 + jump.InstructionsToSkip;
		context.JumpTargets.Add(targetIndex);
		if (jump.InstructionType is InstructionType.JumpIfFalse or InstructionType.JumpIfTrue)
		{
			var condTemp = context.LastConditionTemp ?? "%cond_fallback";
			var fallthroughIndex = currentIndex + 1;
			context.JumpTargets.Add(fallthroughIndex);
			lines.Add(jump.InstructionType == InstructionType.JumpIfFalse
				? $"    cf.cond_br {condTemp}, ^bb{fallthroughIndex}, ^bb{targetIndex}"
				: $"    cf.cond_br {condTemp}, ^bb{targetIndex}, ^bb{fallthroughIndex}");
		}
		else
			lines.Add($"    cf.br ^bb{targetIndex}");
	}

	//ncrunch: no coverage start
	private static void EmitPrint(PrintInstruction print, List<string> lines, EmitContext context)
	{
		var constName = $"@str_{context.FunctionName}_{context.StringConstants.Count}";
		var text = print.TextPrefix + (print.ValueRegister.HasValue
			? "%g\\0A"
			: "\\0A");
		var nullTerminated = text + "\\00";
		var byteLen = CountStringBytes(nullTerminated);
		context.StringConstants.Add((constName, nullTerminated, byteLen));
		if (!print.ValueRegister.HasValue)
		{
			var gepTemp = context.NextTemp();
			lines.Add($"    {gepTemp} = llvm.mlir.addressof {constName}" +
				$" : !llvm.ptr");
			lines.Add($"    %print_{context.TempCounter++} = " +
				$"llvm.call @printf({gepTemp}) {PrintfVarargSignature} : (!llvm.ptr) -> i32");
		}
		else
		{
			var value = context.RegisterValues.GetValueOrDefault(print.ValueRegister.Value, "%zero");
			var gepTemp = context.NextTemp();
			lines.Add($"    {gepTemp} = llvm.mlir.addressof {constName} : !llvm.ptr");
			lines.Add($"    %print_{context.TempCounter++} = " +
				$"llvm.call @printf({gepTemp}, {value}) {PrintfVarargSignature} : (!llvm.ptr, f64) -> i32");
		}
	} //ncrunch: no coverage end

	private const string PrintfVarargSignature = "vararg(!llvm.func<i32 (ptr, ...)>)";

	private static void EmitInvoke(Invoke invoke, List<string> lines, EmitContext context,
		Dictionary<string, CompiledMethodInfo>? compiledMethods)
	{
		if (invoke.MethodInfo == null)
			throw new NotSupportedException( //ncrunch: no coverage
				"Invoke instruction is missing method metadata");
		if (invoke.MethodInfo.MethodName == Method.From && !invoke.MethodInfo.InstanceRegister.HasValue)
		{
			context.RegisterInstances[invoke.Register] = invoke.MethodInfo.ArgumentRegisters;
			return;
		}
		var methodKey = BuildMethodHeaderKeyInternal(invoke.MethodInfo);
		if (compiledMethods == null || !compiledMethods.TryGetValue(methodKey, out var methodInfo))
			throw new NotSupportedException( //ncrunch: no coverage
				//TODO: wtf? why is this still here, support it!
				"Non-print method calls cannot be compiled to MLIR. " +
				"Use the interpreted runner for programs with complex runtime method calls.");
		var arguments = new List<string>();
		if (methodInfo.MemberNames.Count > 0 && invoke.MethodInfo.InstanceRegister.HasValue &&
			context.RegisterInstances.TryGetValue(invoke.MethodInfo.InstanceRegister.Value,
				out var memberRegisters))
			foreach (var reg in memberRegisters)
				arguments.Add(context.RegisterValues.GetValueOrDefault(reg, "0.0"));
		foreach (var argReg in invoke.MethodInfo.ArgumentRegisters)
			arguments.Add(context.RegisterValues.GetValueOrDefault(argReg, "0.0"));
		var constLines = new List<string>();
		var callArgs = new List<string>();
		foreach (var arg in arguments)
		{
			if (arg.StartsWith('%'))
				callArgs.Add(arg); //ncrunch: no coverage
			else
			{
				var constTemp = context.NextTemp();
				constLines.Add($"    {constTemp} = arith.constant {arg} : f64");
				callArgs.Add(constTemp);
			}
		}
		foreach (var constLine in constLines)
			lines.Add(constLine);
		var result = context.NextTemp();
		var argSignature = string.Join(", ", callArgs);
		var typeSignature = string.Join(", ", Enumerable.Repeat("f64", callArgs.Count));
		lines.Add(
			$"    {result} = func.call @{methodInfo.Symbol}({argSignature}) : ({typeSignature}) -> f64");
		context.RegisterValues[invoke.Register] = result;
	}

	private static string BuildEntryPoint(string methodName) =>
		"  func.func @main() -> i32 {\n" +
		$"    %result = func.call @{methodName}() : () -> f64\n" +
		"    %exitCode = arith.constant 0 : i32\n" +
		"    return %exitCode : i32\n" +
		"  }";

	private static void EmitJumpToId(JumpToId jumpToId, List<string> lines, EmitContext context,
		int currentIndex)
	{
		var condTemp = context.LastConditionTemp ?? "%cond_fallback";
		var targetIndex = jumpToId.Id;
		var fallthroughIndex = currentIndex + 1;
		context.JumpTargets.Add(targetIndex);
		context.JumpTargets.Add(fallthroughIndex);
		lines.Add($"    cf.cond_br {condTemp}, ^bb{targetIndex}, ^bb{fallthroughIndex}");
	}

	private static void EmitLoopBegin(LoopBeginInstruction loopBegin, List<string> lines,
		EmitContext context, List<Instruction> instructions, int loopBeginIndex)
	{
		if (!loopBegin.IsRange)
			return;
		var startValue = context.RegisterValues.GetValueOrDefault(loopBegin.Register, "%zero");
		var endValue = context.RegisterValues.GetValueOrDefault(loopBegin.EndIndex!.Value, "%zero");
		var startIndex = context.NextTemp();
		var endIndex = context.NextTemp();
		var step = context.NextTemp();
		var inductionVar = context.NextTemp();
		lines.Add($"    {startIndex} = arith.fptosi {startValue} : f64 to index");
		lines.Add($"    {endIndex} = arith.fptosi {endValue} : f64 to index");
		lines.Add($"    {step} = arith.constant 1 : index");
		var iterationCount = context.RegisterConstants.TryGetValue(loopBegin.EndIndex.Value, out var endConst)
			? (long)endConst
			: 0L;
		var bodyCount = CountLoopBodyInstructions(instructions, loopBeginIndex, loopBegin);
		var complexity = iterationCount * Math.Max(bodyCount, 1);
		context.ActiveLoopCount++;
		if (complexity > GpuComplexityThreshold)
			EmitGpuLaunch(lines, context, startIndex, endIndex);
		else if (complexity > ComplexityThreshold)
			lines.Add($"    scf.parallel ({inductionVar}) = ({startIndex}) to ({endIndex}) step ({step}) {{");
		else
			lines.Add($"    scf.for {inductionVar} = {startIndex} to {endIndex} step {step} {{");
	}

	/// <summary>
	/// Minimum iteration×body-instruction complexity to emit scf.parallel instead of scf.for.
	/// </summary>
	public const int ComplexityThreshold = 100_000;
	/// <summary>
	/// Minimum complexity to offload to GPU via gpu.launch instead of scf.parallel.
	/// </summary>
	public const int GpuComplexityThreshold = 10_000_000;

	private static int CountLoopBodyInstructions(List<Instruction> instructions,
		int loopBeginIndex, LoopBeginInstruction loopBegin)
	{
		var count = 0;
		for (var index = loopBeginIndex + 1; index < instructions.Count; index++)
		{
			if (instructions[index] is LoopEndInstruction loopEnd &&
				ReferenceEquals(loopEnd.Begin, loopBegin))
				return count;
			count++;
		}
		return count;
	}

	private static void EmitGpuLaunch(List<string> lines, EmitContext context,
		string startIndex, string endIndex)
	{
		context.SetGpuActive();
		var numElements = context.NextTemp();
		var hostBuf = context.NextTemp();
		var devBuf = context.NextTemp();
		var gridX = context.NextTemp();
		var gridY = context.NextTemp();
		var gridZ = context.NextTemp();
		var blockY = context.NextTemp();
		var blockZ = context.NextTemp();
		lines.Add($"    {numElements} = arith.subi {endIndex}, {startIndex} : index");
		lines.Add($"    {hostBuf} = memref.alloc({numElements}) : memref<?xf64>");
		lines.Add($"    {devBuf}, %stream = gpu.alloc({numElements}) : memref<?xf64>");
		lines.Add($"    gpu.memcpy %stream {devBuf}, {hostBuf} : memref<?xf64>, memref<?xf64>");
		context.GpuBufferState = new GpuBufferInfo(hostBuf, devBuf);
		lines.Add("    %block_x = arith.constant 256 : index");
		lines.Add($"    {gridX} = arith.ceildivui {numElements}, %block_x : index");
		lines.Add($"    {gridY} = arith.constant 1 : index");
		lines.Add($"    {gridZ} = arith.constant 1 : index");
		lines.Add($"    {blockY} = arith.constant 1 : index");
		lines.Add($"    {blockZ} = arith.constant 1 : index");
		lines.Add($"    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = {gridX}, %grid_y = {gridY}, %grid_z = {gridZ})");
		lines.Add($"               threads(%tx, %ty, %tz) in (%block_x = %block_x, %block_y = {blockY}, %block_z = {blockZ}) {{");
		var globalId = context.NextTemp();
		var blockOffset = context.NextTemp();
		var cond = context.NextTemp();
		lines.Add($"        {blockOffset} = arith.muli %bx, %block_x : index");
		lines.Add($"        {globalId} = arith.addi {blockOffset}, %tx : index");
		lines.Add($"        {cond} = arith.cmpi ult, {globalId}, {numElements} : index");
		lines.Add($"        scf.if {cond} {{");
	}

	private static void EmitLoopEnd(List<string> lines, EmitContext context)
	{
		if (context.ActiveLoopCount == 0)
			return;
		context.ActiveLoopCount--;
		if (context.UsesGpu)
		{
			lines.Add("        }");
			lines.Add("    gpu.terminator");
			lines.Add("    }");
			if (context.GpuBufferState != null)
			{
				lines.Add($"    gpu.dealloc {context.GpuBufferState.DeviceBuffer} : memref<?xf64>");
				lines.Add($"    memref.dealloc {context.GpuBufferState.HostBuffer} : memref<?xf64>");
				context.GpuBufferState = null;
			}
			context.UsesGpu = false;
		}
		else
			lines.Add("    }");
	}

	private static string FormatDouble(double value)
	{
		if (value == 0.0)
			return "0.0";
		var text = value.ToString("G17");
		if (!text.Contains('.') && !text.Contains('E') && !text.Contains('e'))
			text += ".0";
		return text;
	}

	private static int CountStringBytes(string text) =>
		text.Replace("\\0A", "\n").Replace("\\00", "\0").Length;

	private sealed record GpuBufferInfo(string HostBuffer, string DeviceBuffer);

	private sealed class EmitContext(string functionName)
	{
		public string FunctionName { get; } = functionName;
		public string NextTemp() => $"%t{TempCounter++}";
		public int TempCounter;
		public Dictionary<Register, string> RegisterValues { get; } = new();
		public Dictionary<Register, Register[]> RegisterInstances { get; } = new();
		public Dictionary<string, string> VariableValues { get; } = new(StringComparer.Ordinal);
		//unused: public Dictionary<string, Register[]> VariableInstances { get; } = new(StringComparer.Ordinal);
		public Dictionary<string, int> ParamIndexByName { get; } = new(StringComparer.Ordinal);
		public string? LastConditionTemp { get; set; }
		public HashSet<int> JumpTargets { get; } = [];
		public List<(string Name, string Text, int ByteLen)> StringConstants { get; } = [];
		public int ActiveLoopCount;
		public Dictionary<Register, double> RegisterConstants { get; } = new();
		public bool UsesGpu { get; set; }
		public bool HadGpuOps { get; private set; }

		public void SetGpuActive()
		{
			UsesGpu = true;
			HadGpuOps = true;
		}

		public GpuBufferInfo? GpuBufferState { get; set; }
	}
}