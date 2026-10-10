using System.Text;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Assembly;

public sealed partial class InstructionsToMlir
{
	/// <summary>
	/// Jump targets and the instruction after a conditional jump start new blocks (^bb + index).
	/// </summary>
	private static HashSet<int> FindBlockStarts(List<Instruction> instructions,
		Dictionary<int, int> jumpEndIndices)
	{
		var blockStarts = new HashSet<int>();
		for (var index = 0; index < instructions.Count; index++)
			switch (instructions[index])
			{
			case Jump jump:
				blockStarts.Add(index + 1 + jump.InstructionsToSkip);
				if (jump.InstructionType != InstructionType.Jump)
					blockStarts.Add(index + 1);
				break;
			case JumpToId { InstructionType: not InstructionType.JumpEnd } jumpToId:
				blockStarts.Add(FindJumpEnd(jumpToId.Id, jumpEndIndices));
				blockStarts.Add(index + 1);
				break;
			}
		return blockStarts;
	}

	private static int FindJumpEnd(int id, Dictionary<int, int> jumpEndIndices) =>
		jumpEndIndices.TryGetValue(id, out var index)
			? index
			: throw new NotSupportedByBackend("No JumpEnd found for jump id " + id);

	private static void StartBlock(int index, List<string> lines, EmitContext context)
	{
		if (!context.IsTerminated)
			lines.Add($"    cf.br ^bb{index}");
		lines.Add($"  ^bb{index}:");
		context.IsTerminated = false;
	}

	private static void EmitJump(Jump jump, List<string> lines, EmitContext context, int currentIndex)
	{
		var target = $"^bb{currentIndex + 1 + jump.InstructionsToSkip}";
		var fallthrough = $"^bb{currentIndex + 1}";
		lines.Add(jump.InstructionType switch
		{
			InstructionType.JumpIfTrue =>
				$"    cf.cond_br {Condition(lines, context)}, {target}, {fallthrough}",
			InstructionType.JumpIfFalse =>
				$"    cf.cond_br {Condition(lines, context)}, {fallthrough}, {target}",
			InstructionType.JumpIfNotZero =>
				$"    cf.cond_br {IsPositive(((JumpIfNotZero)jump).Register, lines, context)}, " +
				$"{target}, {fallthrough}",
			_ => $"    cf.br {target}"
		});
		context.IsTerminated = true;
	}

	private static string Condition(List<string> lines, EmitContext context)
	{
		if (!context.UsesSlots)
			return context.LastConditionTemp ??
				throw new NotSupportedByBackend("Jump without comparison in " + context.FunctionName);
		var flag = context.NextTemp();
		lines.Add($"    {flag} = llvm.load %flagSlot : !llvm.ptr -> i1");
		return flag;
	}

	private static string IsPositive(Register register, List<string> lines, EmitContext context)
	{
		var zero = context.NextTemp();
		var isPositive = context.NextTemp();
		lines.Add($"    {zero} = arith.constant 0.0 : f64");
		lines.Add($"    {isPositive} = arith.cmpf ogt, {context.Value(register)}, {zero} : f64");
		return isPositive;
	}

	private static void EmitJumpToId(JumpToId jumpToId, List<string> lines, EmitContext context,
		int currentIndex)
	{
		var target = $"^bb{FindJumpEnd(jumpToId.Id, context.JumpEndIndices)}";
		var fallthrough = $"^bb{currentIndex + 1}";
		lines.Add(jumpToId.InstructionType == InstructionType.JumpToIdIfTrue
			? $"    cf.cond_br {Condition(lines, context)}, {target}, {fallthrough}"
			: $"    cf.cond_br {Condition(lines, context)}, {fallthrough}, {target}");
		context.IsTerminated = true;
	}

	private static void EmitPrint(PrintInstruction print, List<string> lines, EmitContext context)
	{
		if (print.ValueRegister.HasValue && (print.ValueIsText ||
			context.BooleanRegisters.Contains(print.ValueRegister.Value)))
			throw new NotSupportedByBackend("MLIR compilation can only print numbers, not the value of " +
				print + " in " + context.FunctionName);
		var constName = $"@str_{context.FunctionName}_{context.StringConstants.Count}";
		var (prefix, prefixLength) = EncodeText(print.TextPrefix);
		var text = prefix + (print.ValueRegister.HasValue
			? "%g\\0A"
			: "\\0A") + "\\00";
		var byteLen = prefixLength + (print.ValueRegister.HasValue
			? 4
			: 2);
		context.StringConstants.Add((constName, text, byteLen));
		var gepTemp = context.NextTemp();
		lines.Add($"    {gepTemp} = llvm.mlir.addressof {constName} : !llvm.ptr");
		var printfCall = $"    %print_{context.TempCounter++} = llvm.call @printf({gepTemp}";
		lines.Add(print.ValueRegister.HasValue
			? $"{printfCall}, {context.Value(print.ValueRegister.Value)}) {PrintfVarargSignature}" +
			" : (!llvm.ptr, f64) -> i32"
			: $"{printfCall}) {PrintfVarargSignature} : (!llvm.ptr) -> i32");
	}

	/// <summary>
	/// MLIR string attributes escape quotes, backslashes and non printable bytes as \XX hex.
	/// </summary>
	private static (string Encoded, int ByteLength) EncodeText(string text)
	{
		var bytes = Encoding.UTF8.GetBytes(text);
		var encoded = new StringBuilder();
		foreach (var textByte in bytes)
			if (textByte is >= 32 and < 127 && textByte != '"' && textByte != '\\')
				encoded.Append((char)textByte);
			else
				encoded.Append('\\').Append(textByte.ToString("X2"));
		return (encoded.ToString(), bytes.Length);
	}

	private static void EmitInvoke(Invoke invoke, List<string> lines, EmitContext context,
		Dictionary<string, CompiledMethodInfo>? compiledMethods)
	{
		var info = invoke.MethodInfo;
		if (info.MethodName == Method.From && !info.InstanceRegister.HasValue)
		{
			context.RegisterInstances[invoke.Register] = ConstructInstance(info, lines, context);
			return;
		}
		var returnType = info.ReturnTypeName.Split(Context.ParentSeparator)[^1];
		if (returnType is not (Type.Number or Type.Boolean or Type.None))
			throw new NotSupportedByBackend("MLIR compilation only supports number results, not " +
				info.TypeFullName + "." + info.MethodName + " returning " + returnType);
		if (compiledMethods == null ||
			!compiledMethods.TryGetValue(BuildMethodHeaderKeyInternal(info), out var methodInfo))
			throw new NotSupportedByBackend("No compiled MLIR function for " + info.TypeFullName + "." +
				info.MethodName + " called in " + context.FunctionName);
		var arguments = new List<string>();
		if (methodInfo.MemberNames.Count > 0)
			arguments.AddRange(info.InstanceRegister.HasValue &&
				context.RegisterInstances.TryGetValue(info.InstanceRegister.Value, out var members)
					? members
					: throw new NotSupportedByBackend("Instance of " + info.TypeFullName + "." +
						info.MethodName + " is not known at compile time in " + context.FunctionName));
		foreach (var argumentRegister in info.ArgumentRegisters)
			arguments.Add(context.Value(argumentRegister));
		var result = context.NextTemp();
		var argumentValues = string.Join(", ", arguments);
		var argumentTypes = string.Join(", ", Enumerable.Repeat("f64", arguments.Count));
		lines.Add($"    {result} = func.call @{methodInfo.Symbol}({argumentValues}) : " +
			$"({argumentTypes}) -> f64");
		context.SetRegister(invoke.Register, result, returnType == Type.Boolean);
	}

	/// <summary>
	/// An instance is the list of its number member values, arguments are matched by member name
	/// like the VM does, missing members use their initial value or 0.
	/// </summary>
	private static List<string> ConstructInstance(InvokeMethodInfo info, List<string> lines,
		EmitContext context)
	{
		var members = GetInstanceMembers(context.Binary, info.TypeFullName);
		if (members.Count == 0)
			return [.. info.ArgumentRegisters.Select(context.Value)];
		var values = new List<string>();
		foreach (var (member, position) in members)
		{
			var argumentIndex = Array.IndexOf(info.ParameterNames, member.Name);
			if (argumentIndex < 0)
				argumentIndex = position;
			if (argumentIndex < info.ArgumentRegisters.Length)
				values.Add(context.Value(info.ArgumentRegisters[argumentIndex]));
			else
			{
				var temp = context.NextTemp();
				var initialValue = member.InitialValueExpression is SetInstruction initial
					? initial.ValueInstance.Number
					: 0;
				lines.Add($"    {temp} = arith.constant {FormatDouble(initialValue)} : f64");
				values.Add(temp);
			}
		}
		return values;
	}
}
