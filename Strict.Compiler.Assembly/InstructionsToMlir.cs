using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
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
	//TODO: clean up!
	/// <summary>Minimum iteration×body-instruction complexity to emit scf.parallel instead of scf.for.</summary>
	public const int ComplexityThreshold = 100_000;
	/// <summary>Minimum complexity to offload to GPU via gpu.launch instead of scf.parallel.</summary>
	public const int GpuComplexityThreshold = 10_000_000;

	public override Task<string> Compile(BinaryExecutable binary, Platform platform)
	{
   var output = CompileForPlatform(Method.Run, binary, platform);
		return Task.FromResult(output);
	}

	public override string Extension => ".mlir";

	//TODO: duplicated code, should be in base or removed!
	private sealed class CompiledMethodInfo(string symbol,
		List<Instruction> instructions, List<string> parameterNames, List<string> memberNames)
	{
		public string Symbol { get; } = symbol;
		public List<Instruction> Instructions { get; } = instructions;
		public List<string> ParameterNames { get; } = parameterNames;
		public List<string> MemberNames { get; } = memberNames;
	}

	public string CompileInstructions(string methodName, List<Instruction> instructions) =>
		BuildFunction(methodName, [], instructions).Text;

	public string CompileForPlatform(string methodName, BinaryExecutable binary, Platform platform,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods = null) =>
   CompileForPlatform(methodName, binary.EntryPoint.instructions, platform,
			precompiledMethods ?? BuildPrecompiledMethodsInternal(binary));

	public string CompileForPlatform(string methodName, IReadOnlyList<Instruction> instructions,
		Platform platform, IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods = null)
	{
		var hasPrint = instructions.OfType<PrintInstruction>().Any();
		var methodInfos = CollectMethods([.. instructions], precompiledMethods);
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

	public bool HasPrintInstructions(IReadOnlyList<Instruction> instructions) =>
		HasPrintInstructionsInternal(instructions);

	public bool IsPlatformUsingStdLibAndHasPrintInstructions(Platform platform,
		IReadOnlyList<Instruction> optimizedInstructions,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods) =>
		IsPlatformUsingStdLibAndHasPrintInstructionsInternal(platform, optimizedInstructions,
			precompiledMethods, includeWindowsPlatform: true);

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
		switch (instruction)
		{
		case LoadConstantInstruction loadConst:
			EmitLoadConstant(loadConst, lines, context);
			break;
		case BinaryInstruction binary:
			EmitBinary(binary, lines, context);
			break;
		case ReturnInstruction ret:
			EmitReturn(ret, lines, context);
			break;
		case StoreFromRegisterInstruction storeReg:
			EmitStoreFromRegister(storeReg, context);
			break;
		case LoadVariableToRegister loadVar:
			EmitLoadVariable(loadVar, context);
			break;
		case StoreVariableInstruction storeVar:
			EmitStoreVariable(storeVar, context);
			break;
		case Jump jump:
			EmitJump(jump, lines, context, index);
			break;
		case PrintInstruction print:
			EmitPrint(print, lines, context); //ncrunch: no coverage
			break; //ncrunch: no coverage
		case Invoke invoke:
			EmitInvoke(invoke, lines, context, compiledMethods);
			break;
		case JumpToId jumpToId:
			EmitJumpToId(jumpToId, lines, context, index); //ncrunch: no coverage
			break; //ncrunch: no coverage
		case LoopBeginInstruction loopBegin:
			EmitLoopBegin(loopBegin, lines, context, instructions, index);
			break;
		case LoopEndInstruction:
			EmitLoopEnd(lines, context);
			break;
		default:
			throw new NotSupportedException(
				$"MLIR compilation does not support instruction: {instruction.GetType().Name} ({instruction.InstructionType})");
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
		if (context.RegisterInstances.TryGetValue(storeReg.Register, out var instances))
			context.VariableInstances[storeReg.Identifier] = instances;
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
		if (invoke.Method == null)
			throw new NotSupportedException( //ncrunch: no coverage
				"Invoke instruction is missing method metadata");
		if (invoke.Method.Method.Name == Method.From && invoke.Method.Instance == null)
		{
			context.RegisterInstances[invoke.Register] = ResolveConstructorArguments(invoke.Method);
			return;
		}
		var methodKey = BuildMethodHeaderKeyInternal(invoke.Method.Method);
		if (compiledMethods == null || !compiledMethods.TryGetValue(methodKey, out var methodInfo))
			throw new NotSupportedException( //ncrunch: no coverage
				//TODO: wtf?
				"Non-print method calls cannot be compiled to MLIR. " +
				"Use the interpreted runner for programs with complex runtime method calls.");
		var arguments = new List<string>();
		if (methodInfo.MemberNames.Count > 0)
			foreach (var memberExpression in ResolveInstanceMemberArguments(invoke.Method,
				context.VariableInstances))
				arguments.Add(ResolveExpressionValue(memberExpression, context));
		for (var argIndex = 0; argIndex < invoke.Method.Arguments.Count; argIndex++)
			arguments.Add(ResolveExpressionValue(invoke.Method.Arguments[argIndex], context));
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

	private static Dictionary<string, CompiledMethodInfo> CollectMethods(
		List<Instruction> instructions,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods)
	{
		var methods = new Dictionary<string, CompiledMethodInfo>(StringComparer.Ordinal);
   var queue = new Queue<(Method Method, bool IncludeMembers)>();
		EnqueueInvokedMethods(instructions, queue);
		while (queue.Count > 0)
		{
			var (method, includeMembers) = queue.Dequeue();
			var methodKey = BuildMethodHeaderKeyInternal(method);
			if (methods.TryGetValue(methodKey, out var existing))
			{
				if (includeMembers && existing.MemberNames.Count == 0)
					methods[methodKey] = BuildMethodInfo(method, true, precompiledMethods);
				continue;
			}
			var methodInfo = BuildMethodInfo(method, includeMembers, precompiledMethods);
			methods[methodKey] = methodInfo;
			EnqueueInvokedMethods(methodInfo.Instructions, queue);
		}
		return methods;
	}

	private static CompiledMethodInfo BuildMethodInfo(Method method, bool includeMembers,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods)
	{
		var methodKey = BuildMethodHeaderKeyInternal(method);
		var instructions =
			precompiledMethods != null && precompiledMethods.TryGetValue(methodKey, out var precompiled)
				? [.. precompiled]
				: GenerateInstructions(method);
		var memberNames = includeMembers
			? method.Type.Members.Where(member => !member.Type.IsTrait).Select(member => member.Name).ToList()
			: new List<string>();
		var parameterNames = new List<string>(memberNames);
		parameterNames.AddRange(method.Parameters.Select(parameter => parameter.Name));
		return new CompiledMethodInfo(BuildMethodSymbol(method), instructions, parameterNames,
			memberNames);
	}

	private static void EnqueueInvokedMethods(IEnumerable<Instruction> instructions,
		Queue<(Method Method, bool IncludeMembers)> queue)
	{
		foreach (var invoke in instructions.OfType<Invoke>())
			if (invoke.Method != null && invoke.Method.Method.Name != Method.From)
				queue.Enqueue((invoke.Method.Method, invoke.Method.Instance != null));
	}

	private static string BuildMethodSymbol(Method method) =>
		method.Type.Name + "_" + method.Name + "_" + method.Parameters.Count;

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
		context.LoopStack.Push(new LoopState(startIndex, endIndex, step, inductionVar));
		if (complexity > GpuComplexityThreshold)
			EmitGpuLaunch(lines, context, startIndex, endIndex, step, inductionVar);
		else if (complexity > ComplexityThreshold)
			lines.Add($"    scf.parallel ({inductionVar}) = ({startIndex}) to ({endIndex}) step ({step}) {{");
		else
			lines.Add($"    scf.for {inductionVar} = {startIndex} to {endIndex} step {step} {{");
	}

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
		string startIndex, string endIndex, string step, string inductionVar)
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
		context.GpuBufferState = new GpuBufferInfo(hostBuf, devBuf, numElements);
		lines.Add($"    %block_x = arith.constant 256 : index");
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
		if (context.LoopStack.Count == 0)
			return;
		context.LoopStack.Pop();
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

	private static List<Expression> ResolveConstructorArguments(MethodCall constructorCall)
	{
		var members = constructorCall.ReturnType.Members.Where(member => !member.Type.IsTrait).ToList();
		var result = new List<Expression>(members.Count);
		for (var index = 0; index < members.Count; index++)
			result.Add(index < constructorCall.Arguments.Count
				? constructorCall.Arguments[index]
				: new Value(members[index].Type, new ValueInstance(members[index].Type, 0)));
		return result;
	}

	private static IEnumerable<Expression> ResolveInstanceMemberArguments(MethodCall methodCall,
		Dictionary<string, List<Expression>> variableInstances)
	{
		if (methodCall.Instance is MethodCall constructorCall &&
			constructorCall.Method.Name == Method.From && constructorCall.Instance == null)
			return ResolveConstructorArguments(constructorCall);
		var instanceName = methodCall.Instance?.ToString();
		if (instanceName != null && variableInstances.TryGetValue(instanceName, out var values))
			return values;
		throw new NotSupportedException("Cannot resolve instance values for method call: " + methodCall);
	}

	private static string ResolveExpressionValue(Expression expression, EmitContext context)
	{
		if (expression is Value value && !value.Data.IsText)
			return FormatDouble(value.Data.Number);
		var variableName = expression.ToString();
		if (context.ParamIndexByName.TryGetValue(variableName, out var paramIndex))
			return $"%param{paramIndex}";
		if (context.VariableValues.TryGetValue(variableName, out var variableValue))
			return variableValue;
		throw new NotSupportedException("Unsupported expression for MLIR compilation: " + expression);
	}

	private sealed record LoopState(string StartIndex, string EndIndex, string Step,
		string InductionVar);

	private sealed record GpuBufferInfo(string HostBuffer, string DeviceBuffer,
		string NumElements);

	private static List<Instruction> GenerateInstructions(Method method) =>
		throw new NotSupportedException("Method fallback instruction generation is not supported. Use BinaryExecutable entry-point/precompiled methods.");

	private sealed class EmitContext(string functionName)
	{
		public string FunctionName { get; } = functionName;
		public string NextTemp() => $"%t{TempCounter++}";
		public int TempCounter;
		public Dictionary<Register, string> RegisterValues { get; } = new();
		public Dictionary<Register, List<Expression>> RegisterInstances { get; } = new();
		public Dictionary<string, string> VariableValues { get; } = new(StringComparer.Ordinal);
		public Dictionary<string, List<Expression>> VariableInstances { get; } =
			new(StringComparer.Ordinal);
		public Dictionary<string, int> ParamIndexByName { get; } = new(StringComparer.Ordinal);
		public string? LastConditionTemp { get; set; }
		public HashSet<int> JumpTargets { get; } = [];
		public List<(string Name, string Text, int ByteLen)> StringConstants { get; } = [];
		public Stack<LoopState> LoopStack { get; } = new();
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