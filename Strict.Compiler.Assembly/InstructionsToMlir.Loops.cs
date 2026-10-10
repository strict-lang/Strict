using Strict.Bytecode.Instructions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Assembly;

public sealed partial class InstructionsToMlir
{
	private sealed record LoopState(int Id, bool IsStructured, List<(string Slot, string Value)> Saved);

	private static string LoopCounterName(int loopBeginIndex) => "#loop" + loopBeginIndex;

	private static IEnumerable<string> LoopVariableNames(LoopBeginInstruction loopBegin) =>
		new[] { Type.IndexLowercase, Type.ValueLowercase }.Concat(loopBegin.CustomVariableNames);

	private static void EmitLoopBegin(LoopBeginInstruction loopBegin, List<string> lines,
		EmitContext context, List<Instruction> instructions, int loopBeginIndex)
	{
		var body = GetLoopBody(instructions, loopBeginIndex);
		if (IsStructuredRange(loopBegin, body, context))
			EmitStructuredLoopBegin(loopBegin, lines, context, body);
		else
			EmitBranchingLoopBegin(loopBegin, lines, context, loopBeginIndex);
	}

	private static List<Instruction> GetLoopBody(List<Instruction> instructions, int loopBeginIndex)
	{
		var depth = 0;
		for (var index = loopBeginIndex + 1; index < instructions.Count; index++)
			if (instructions[index] is LoopBeginInstruction)
				depth++;
			else if (instructions[index] is LoopEndInstruction && depth-- == 0)
				return instructions.GetRange(loopBeginIndex + 1, index - loopBeginIndex - 1);
		throw new NotSupportedByBackend("LoopBegin at " + loopBeginIndex + " has no LoopEnd");
	}

	/// <summary>
	/// scf.for, scf.parallel and gpu.launch need a single block body counting up between constants.
	/// </summary>
	private static bool IsStructuredRange(LoopBeginInstruction loopBegin,
		IEnumerable<Instruction> body, EmitContext context) =>
		loopBegin.IsRange &&
		context.RegisterConstants.TryGetValue(loopBegin.Register, out var start) &&
		context.RegisterConstants.TryGetValue(loopBegin.EndIndex!.Value, out var end) && start <= end &&
		!body.Any(instruction => instruction is Jump or JumpToId or ReturnInstruction or
			LoopBeginInstruction);

	private static void EmitStructuredLoopBegin(LoopBeginInstruction loopBegin, List<string> lines,
		EmitContext context, List<Instruction> body)
	{
		var startIndex = context.NextTemp();
		var endIndex = context.NextTemp();
		var step = context.NextTemp();
		var inductionVar = context.NextTemp();
		lines.Add($"    {startIndex} = arith.fptosi {context.Value(loopBegin.Register)} : f64 to index");
		lines.Add($"    {endIndex} = arith.fptosi {context.Value(loopBegin.EndIndex!.Value)} : f64 to index");
		lines.Add($"    {step} = arith.constant 1 : index");
		var complexity = (long)(context.RegisterConstants[loopBegin.EndIndex.Value] -
			context.RegisterConstants[loopBegin.Register]) * Math.Max(body.Count, 1);
		var isParallel = !body.Any(instruction =>
			instruction is StoreFromRegisterInstruction or StoreVariableInstruction);
		context.Loops.Push(new LoopState(context.Loops.Count, true, []));
		if (isParallel && complexity > GpuComplexityThreshold)
			inductionVar = EmitGpuLaunch(lines, context, startIndex, endIndex);
		else if (isParallel && complexity > ComplexityThreshold)
			lines.Add($"    scf.parallel ({
				inductionVar
			}) = ({
				startIndex
			}) to ({
				endIndex
			}) step ({
				step
			}) {{");
		else
			lines.Add($"    scf.for {inductionVar} = {startIndex} to {endIndex} step {step} {{");
		var integer = context.NextTemp();
		var number = context.NextTemp();
		lines.Add($"    {integer} = arith.index_cast {inductionVar} : index to i64");
		lines.Add($"    {number} = arith.sitofp {integer} : i64 to f64");
		foreach (var name in LoopVariableNames(loopBegin))
			context.LoopValues[name] = number;
	}

	/// <summary>
	/// Counts like the VM: Range(start, end) steps by one towards end (exclusive), a number N runs
	/// N times, index and value are the current position and restored after the loop.
	/// </summary>
	private static void EmitBranchingLoopBegin(LoopBeginInstruction loopBegin, List<string> lines,
		EmitContext context, int loopBeginIndex)
	{
		var (start, count, step) = loopBegin.IsRange
			? EmitRangeBounds(loopBegin, lines, context)
			: EmitCountBounds(loopBegin, lines, context);
		var saved = new List<(string Slot, string Value)>();
		foreach (var name in LoopVariableNames(loopBegin))
		{
			var value = context.NextTemp();
			lines.Add($"    {value} = llvm.load {context.Slots[name]} : !llvm.ptr -> f64");
			saved.Add((context.Slots[name], value));
		}
		var counter = context.Slots[LoopCounterName(loopBeginIndex)];
		lines.Add($"    llvm.store %slotZero, {counter} : f64, !llvm.ptr");
		lines.Add($"    cf.br ^loop{loopBeginIndex}");
		lines.Add($"  ^loop{loopBeginIndex}:");
		var iteration = context.NextTemp();
		var hasMore = context.NextTemp();
		lines.Add($"    {iteration} = llvm.load {counter} : !llvm.ptr -> f64");
		lines.Add($"    {hasMore} = arith.cmpf olt, {iteration}, {count} : f64");
		lines.Add($"    cf.cond_br {hasMore}, ^loop{loopBeginIndex}body, ^loop{loopBeginIndex}exit");
		lines.Add($"  ^loop{loopBeginIndex}body:");
		var offset = context.NextTemp();
		var current = context.NextTemp();
		lines.Add($"    {offset} = arith.mulf {iteration}, {step} : f64");
		lines.Add($"    {current} = arith.addf {start}, {offset} : f64");
		foreach (var (slot, _) in saved)
			lines.Add($"    llvm.store {current}, {slot} : f64, !llvm.ptr");
		context.Loops.Push(new LoopState(loopBeginIndex, false, saved));
	}

	private static (string Start, string Count, string Step) EmitRangeBounds(
		LoopBeginInstruction loopBegin, List<string> lines, EmitContext context)
	{
		var start = context.Value(loopBegin.Register);
		var end = context.Value(loopBegin.EndIndex!.Value);
		var (isDecreasing, upwards, downwards, count) =
			(context.NextTemp(), context.NextTemp(), context.NextTemp(), context.NextTemp());
		var (one, minusOne, step) = (context.NextTemp(), context.NextTemp(), context.NextTemp());
		lines.Add($"    {isDecreasing} = arith.cmpf olt, {end}, {start} : f64");
		lines.Add($"    {upwards} = arith.subf {end}, {start} : f64");
		lines.Add($"    {downwards} = arith.subf {start}, {end} : f64");
		lines.Add($"    {count} = arith.select {isDecreasing}, {downwards}, {upwards} : f64");
		lines.Add($"    {one} = arith.constant 1.0 : f64");
		lines.Add($"    {minusOne} = arith.constant -1.0 : f64");
		lines.Add($"    {step} = arith.select {isDecreasing}, {minusOne}, {one} : f64");
		return (start, count, step);
	}

	private static (string Start, string Count, string Step) EmitCountBounds(
		LoopBeginInstruction loopBegin, List<string> lines, EmitContext context)
	{
		var (start, truncated, count, step) =
			(context.NextTemp(), context.NextTemp(), context.NextTemp(), context.NextTemp());
		lines.Add($"    {start} = arith.constant 0.0 : f64");
		lines.Add($"    {truncated} = arith.fptosi {context.Value(loopBegin.Register)} : f64 to i64");
		lines.Add($"    {count} = arith.sitofp {truncated} : i64 to f64");
		lines.Add($"    {step} = arith.constant 1.0 : f64");
		return (start, count, step);
	}

	private static string EmitGpuLaunch(List<string> lines, EmitContext context, string startIndex,
		string endIndex)
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
		lines.Add($"    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = {
			gridX
		}, %grid_y = {
			gridY
		}, %grid_z = {
			gridZ
		})");
		lines.Add($"               threads(%tx, %ty, %tz) in (%block_x = %block_x, %block_y = {
			blockY
		}, %block_z = {
			blockZ
		}) {{");
		var globalId = context.NextTemp();
		var blockOffset = context.NextTemp();
		var cond = context.NextTemp();
		var position = context.NextTemp();
		lines.Add($"        {blockOffset} = arith.muli %bx, %block_x : index");
		lines.Add($"        {globalId} = arith.addi {blockOffset}, %tx : index");
		lines.Add($"        {cond} = arith.cmpi ult, {globalId}, {numElements} : index");
		lines.Add($"        scf.if {cond} {{");
		lines.Add($"        {position} = arith.addi {startIndex}, {globalId} : index");
		return position;
	}

	private static void EmitLoopEnd(List<string> lines, EmitContext context)
	{
		if (context.Loops.Count == 0)
			throw new NotSupportedByBackend("LoopEnd without LoopBegin in " + context.FunctionName);
		var loop = context.Loops.Pop();
		if (loop.IsStructured)
			EmitStructuredLoopEnd(lines, context);
		else
			EmitBranchingLoopEnd(loop, lines, context);
	}

	private static void EmitStructuredLoopEnd(List<string> lines, EmitContext context)
	{
		context.LoopValues.Clear();
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
		{
			lines.Add("    }");
		}
	}

	private static void EmitBranchingLoopEnd(LoopState loop, List<string> lines, EmitContext context)
	{
		var counter = context.Slots[LoopCounterName(loop.Id)];
		var (iteration, one, next) = (context.NextTemp(), context.NextTemp(), context.NextTemp());
		lines.Add($"    {iteration} = llvm.load {counter} : !llvm.ptr -> f64");
		lines.Add($"    {one} = arith.constant 1.0 : f64");
		lines.Add($"    {next} = arith.addf {iteration}, {one} : f64");
		lines.Add($"    llvm.store {next}, {counter} : f64, !llvm.ptr");
		lines.Add($"    cf.br ^loop{loop.Id}");
		lines.Add($"  ^loop{loop.Id}exit:");
		foreach (var (slot, value) in loop.Saved)
			lines.Add($"    llvm.store {value}, {slot} : f64, !llvm.ptr");
		context.IsTerminated = false;
	}

	private sealed record GpuBufferInfo(string HostBuffer, string DeviceBuffer);
}
