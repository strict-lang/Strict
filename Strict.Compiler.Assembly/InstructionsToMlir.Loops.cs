using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Language;

namespace Strict.Compiler.Assembly;

public sealed partial class InstructionsToMlir
{
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
		var iterationCount =
			context.RegisterConstants.TryGetValue(loopBegin.EndIndex.Value, out var endConst)
				? (long)endConst
				: 0L;
		var bodyCount = CountLoopBodyInstructions(instructions, loopBeginIndex, loopBegin);
		var complexity = iterationCount * Math.Max(bodyCount, 1);
		context.ActiveLoopCount++;
		if (complexity > GpuComplexityThreshold)
			EmitGpuLaunch(lines, context, startIndex, endIndex);
		else if (complexity > ComplexityThreshold)
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
	}

	private static int CountLoopBodyInstructions(List<Instruction> instructions, int loopBeginIndex,
		LoopBeginInstruction loopBegin)
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

	private static void EmitGpuLaunch(List<string> lines, EmitContext context, string startIndex,
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
		{
			lines.Add("    }");
		}
	}

	private sealed record GpuBufferInfo(string HostBuffer, string DeviceBuffer);
}
