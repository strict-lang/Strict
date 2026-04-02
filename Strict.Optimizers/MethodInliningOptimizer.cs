using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Optimizers;

public sealed class MethodInliningOptimizer : InstructionOptimizer
{
	public override void Optimize(Bytecode.BinaryExecutable binary)
	{
		foreach (var typeEntry in binary.MethodsPerType)
		foreach (var methodGroup in typeEntry.Value.MethodGroups.Values)
		foreach (var method in methodGroup)
			method.instructions = InlineInstructions(binary, typeEntry.Key, method.Name,
				method.instructions);
	}

	public override List<Instruction> Optimize(List<Instruction> instructions) => instructions;

	private List<Instruction> InlineInstructions(Bytecode.BinaryExecutable binary,
		string currentTypeName, string currentMethodName, List<Instruction> instructions)
	{
		var optimized = new List<Instruction>(instructions.Count);
		foreach (var instruction in instructions)
			if (instruction is Invoke invoke && TryInline(binary, currentTypeName, currentMethodName,
				invoke, out var inlinedInstructions))
				optimized.AddRange(inlinedInstructions);
			else
				optimized.Add(instruction);
		return optimized;
	}

	private bool TryInline(Bytecode.BinaryExecutable binary, string currentTypeName,
		string currentMethodName, Invoke invoke, out List<Instruction> inlinedInstructions)
	{
		inlinedInstructions = [];
		if (!CanInline(binary, currentTypeName, currentMethodName, invoke))
			return false;
		var generatedInstructions =
			Bytecode.BinaryGenerator.GenerateInlineInstructions(binary.basePackage,
				GetInlinedExpression(invoke.Method));
		if (!IsInlineBlock(generatedInstructions))
			return false;
		var recursivelyInlinedInstructions = InlineInstructions(binary, currentTypeName,
			currentMethodName, generatedInstructions);
		if (!IsInlineBlock(recursivelyInlinedInstructions))
			return false;
		inlinedInstructions = RemapRegisters(recursivelyInlinedInstructions, invoke.Register);
		return true;
	}

	private static bool CanInline(Bytecode.BinaryExecutable binary, string currentTypeName,
		string currentMethodName, Invoke invoke)
	{
		if (invoke.Method.Instance != null || invoke.Method.Method.Name == currentMethodName ||
			!IsCurrentTypeCall(currentTypeName, invoke.Method.Method.Type.FullName,
				invoke.Method.Method.Type.Name))
			return false;
		var compiledMethod = FindCompiledMethod(binary, currentTypeName, invoke.Method.Method);
		return compiledMethod != null && IsInlineBlock(compiledMethod.instructions);
	}

	private static bool IsCurrentTypeCall(string currentTypeName, string invokedTypeFullName,
		string invokedTypeName) =>
		invokedTypeFullName == currentTypeName || invokedTypeName == currentTypeName ||
		currentTypeName.EndsWith(Context.ParentSeparator + invokedTypeName,
			StringComparison.Ordinal);

	private static BinaryMethod? FindCompiledMethod(Bytecode.BinaryExecutable binary,
		string currentTypeName, Method method) =>
		binary.MethodsPerType.TryGetValue(currentTypeName, out var typeData)
			? typeData.MethodGroups.GetValueOrDefault(method.Name)?.FirstOrDefault(candidate =>
				candidate.parameters.Count == method.Parameters.Count &&
				candidate.ReturnTypeName.EndsWith(method.ReturnType.Name, StringComparison.Ordinal))
			: null;

	private static bool IsInlineBlock(IReadOnlyList<Instruction> instructions)
	{
		if (instructions.Count == 0 || instructions[^1] is not ReturnInstruction)
			return false;
		for (var index = 0; index < instructions.Count - 1; index++)
			if (instructions[index] is not (LoadVariableToRegister or LoadConstantInstruction
				or BinaryInstruction or ListCallInstruction or Invoke or SetInstruction))
				return false;
		return true;
	}

	private static Expression GetInlinedExpression(MethodCall call)
	{
		var body = call.Method.GetBodyAndParseIfNeeded();
		var implementation = body is Body methodBody
			? methodBody.Expressions[^1]
			: body;
		return SubstituteParameters(implementation, call.Method.Parameters, call.Arguments);
	}

	private static Expression SubstituteParameters(Expression expression,
		IReadOnlyList<Parameter> parameters, IReadOnlyList<Expression> arguments)
	{
		if (expression is ParameterCall parameterCall)
			for (var index = 0; index < parameters.Count && index < arguments.Count; index++)
				if (parameters[index].Name == parameterCall.Parameter.Name)
					return arguments[index];
		return expression switch
		{
			Binary binary => new Binary(SubstituteParameters(binary.Instance!, parameters, arguments),
				binary.Method, [SubstituteParameters(binary.Arguments[0], parameters, arguments)]),
			Not not => new Not(not.Method, SubstituteParameters(not.Instance!, parameters, arguments)),
			MethodCall methodCall => new MethodCall(methodCall.Method, methodCall.Instance != null
					? SubstituteParameters(methodCall.Instance, parameters, arguments)
					: null,
				methodCall.Arguments.Select(argument =>
					SubstituteParameters(argument, parameters, arguments)).ToArray()),
			MemberCall memberCall => new MemberCall(memberCall.Instance != null
				? SubstituteParameters(memberCall.Instance, parameters, arguments)
				: null, memberCall.Member, memberCall.LineNumber),
			ListCall listCall => new ListCall(
				SubstituteParameters(listCall.List, parameters, arguments),
				SubstituteParameters(listCall.Index, parameters, arguments), listCall.SecondIndex != null
					? SubstituteParameters(listCall.SecondIndex, parameters, arguments)
					: null, SubstituteParameters(listCall.OriginalIndex, parameters, arguments)),
			If ifExpression => new If(
				SubstituteParameters(ifExpression.Condition, parameters, arguments),
				SubstituteParameters(ifExpression.Then, parameters, arguments), ifExpression.LineNumber,
				ifExpression.OptionalElse != null
					? SubstituteParameters(ifExpression.OptionalElse, parameters, arguments)
					: null),
			_ => expression
		};
	}

	private static List<Instruction> RemapRegisters(IReadOnlyList<Instruction> instructions,
		Bytecode.Register targetRegister)
	{
		var returnRegister = ((ReturnInstruction)instructions[^1]).Register;
		var map = new Dictionary<Bytecode.Register, Bytecode.Register>
		{
			[returnRegister] = targetRegister
		};
		var nextRegister = ((int)targetRegister + 1) %
			Enum.GetValues<Bytecode.Register>().Length;
		foreach (var register in instructions.SelectMany(GetRegisters))
			if (!map.ContainsKey(register))
			{
				while (map.ContainsValue((Bytecode.Register)nextRegister))
					nextRegister = (nextRegister + 1) %
						Enum.GetValues<Bytecode.Register>().Length;
				map[register] = (Bytecode.Register)nextRegister;
				nextRegister = (nextRegister + 1) %
					Enum.GetValues<Bytecode.Register>().Length;
			}
		var remapped = new List<Instruction>(instructions.Count - 1);
		for (var index = 0; index < instructions.Count - 1; index++)
			remapped.Add(Clone(instructions[index], map));
		return remapped;
	}

	private static IEnumerable<Bytecode.Register> GetRegisters(Instruction instruction) =>
		instruction switch
		{
			ListCallInstruction listCall => [listCall.Register, listCall.IndexValueRegister],
			BinaryInstruction binary => binary.Registers,
			RegisterInstruction registerInstruction => [registerInstruction.Register],
			_ => []
		};

	private static Instruction Clone(Instruction instruction,
		IReadOnlyDictionary<Bytecode.Register, Bytecode.Register> registerMap) =>
		instruction switch
		{
			LoadVariableToRegister loadVariable => new LoadVariableToRegister(
				registerMap[loadVariable.Register], loadVariable.Identifier),
			LoadConstantInstruction loadConstant => new LoadConstantInstruction(
				registerMap[loadConstant.Register], loadConstant.Constant),
			BinaryInstruction binary => new BinaryInstruction(binary.InstructionType,
				binary.Registers.Select(register => registerMap[register]).ToArray()),
			ListCallInstruction listCall => new ListCallInstruction(registerMap[listCall.Register],
				registerMap[listCall.IndexValueRegister], listCall.Identifier),
			Invoke nestedInvoke => new Invoke(registerMap[nestedInvoke.Register], nestedInvoke.Method,
				nestedInvoke.PersistedRegistry),
			SetInstruction set => new SetInstruction(set.ValueInstance, registerMap[set.Register]),
			_ => instruction
		};
}