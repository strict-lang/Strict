using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Language;

namespace Strict.Compiler.Assembly;

public sealed partial class InstructionsToMlir
{
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
		{
			lines.Add($"    cf.br ^bb{targetIndex}");
		}
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
			lines.Add($"    {gepTemp} = llvm.mlir.addressof {constName}" + $" : !llvm.ptr");
			lines.Add($"    %print_{context.TempCounter++} = " + $"llvm.call @printf({
				gepTemp
			}) {
				PrintfVarargSignature
			} : (!llvm.ptr) -> i32");
		}
		else
		{
			var value = context.RegisterValues.GetValueOrDefault(print.ValueRegister.Value, "%zero");
			var gepTemp = context.NextTemp();
			lines.Add($"    {gepTemp} = llvm.mlir.addressof {constName} : !llvm.ptr");
			lines.Add($"    %print_{context.TempCounter++} = " + $"llvm.call @printf({
				gepTemp
			}, {
				value
			}) {
				PrintfVarargSignature
			} : (!llvm.ptr, f64) -> i32");
		}
	} //ncrunch: no coverage end

	private static void EmitInvoke(Invoke invoke, List<string> lines, EmitContext context,
		Dictionary<string, CompiledMethodInfo>? compiledMethods)
	{
		if (invoke.MethodInfo == null)
			throw new NotSupportedByBackend( //ncrunch: no coverage
				"Invoke instruction is missing method metadata");
		if (invoke.MethodInfo.MethodName == Method.From && !invoke.MethodInfo.InstanceRegister.HasValue)
		{
			context.RegisterInstances[invoke.Register] = invoke.MethodInfo.ArgumentRegisters;
			return;
		}
		var methodKey = BuildMethodHeaderKeyInternal(invoke.MethodInfo);
		if (compiledMethods == null || !compiledMethods.TryGetValue(methodKey, out var methodInfo))
			throw new NotSupportedByBackend( //ncrunch: no coverage
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
			if (arg.StartsWith('%'))
			{
				callArgs.Add(arg);
			}
			//ncrunch: no coverage
			else
			{
				var constTemp = context.NextTemp();
				constLines.Add($"    {constTemp} = arith.constant {arg} : f64");
				callArgs.Add(constTemp);
			}
		foreach (var constLine in constLines)
			lines.Add(constLine);
		var result = context.NextTemp();
		var argSignature = string.Join(", ", callArgs);
		var typeSignature = string.Join(", ", Enumerable.Repeat("f64", callArgs.Count));
		lines.Add($"    {
			result
		} = func.call @{
			methodInfo.Symbol
		}({
			argSignature
		}) : ({
			typeSignature
		}) -> f64");
		context.RegisterValues[invoke.Register] = result;
	}

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
}
