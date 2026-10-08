using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;

namespace Strict.Optimizers;

public sealed class MethodInliningOptimizer : InstructionOptimizer
{
	public override void Optimize(BinaryExecutable binary)
	{
		foreach (var typeEntry in binary.MethodsPerType)
		foreach (var methodGroup in typeEntry.Value.MethodGroups.Values)
		foreach (var method in methodGroup)
			method.instructions = InlineInstructions(binary, typeEntry.Key, method.Name,
				method.instructions);
	}

	public override List<Instruction> Optimize(List<Instruction> instructions) => instructions;

	private List<Instruction> InlineInstructions(BinaryExecutable binary, string currentTypeName,
		string currentMethodName, List<Instruction> instructions, HashSet<string>? beingInlined = null)
	{
		var optimized = new List<Instruction>(instructions.Count);
		foreach (var instruction in instructions)
			if (instruction is Invoke invoke && TryInline(binary, currentTypeName, currentMethodName,
				invoke, out var inlinedInstructions, beingInlined ??= []))
				optimized.AddRange(inlinedInstructions);
			else
				optimized.Add(instruction);
		return optimized;
	}

	private bool TryInline(BinaryExecutable binary, string currentTypeName, string currentMethodName,
		Invoke invoke, out List<Instruction> inlinedInstructions, HashSet<string> beingInlined)
	{
		inlinedInstructions = [];
		if (!CanInline(binary, currentTypeName, currentMethodName, invoke))
			return false;
		var compiledMethod = FindCompiledMethod(binary, currentTypeName, invoke.MethodInfo);
		if (compiledMethod == null || !IsInlineBlock(compiledMethod.instructions))
			return false;
		if (!beingInlined.Add(compiledMethod.Name))
			return false;
		var recursivelyInlinedInstructions = InlineInstructions(binary, currentTypeName,
			currentMethodName, compiledMethod.instructions, beingInlined);
		beingInlined.Remove(compiledMethod.Name);
		if (!IsInlineBlock(recursivelyInlinedInstructions))
			return false;
		inlinedInstructions = RemapInstructions(recursivelyInlinedInstructions, compiledMethod, invoke);
		return inlinedInstructions.Count > 0;
	}

	private static bool CanInline(BinaryExecutable binary, string currentTypeName,
		string currentMethodName, Invoke invoke)
	{
		if (invoke.MethodInfo.InstanceRegister.HasValue ||
			invoke.MethodInfo.MethodName == currentMethodName ||
			!IsCurrentTypeCall(currentTypeName, invoke.MethodInfo.TypeFullName))
			return false;
		var compiledMethod = FindCompiledMethod(binary, currentTypeName, invoke.MethodInfo);
		return compiledMethod != null && IsInlineBlock(compiledMethod.instructions);
	}

	private static bool IsCurrentTypeCall(string currentTypeName, string invokedTypeFullName) =>
		invokedTypeFullName == currentTypeName ||
		currentTypeName.EndsWith(Context.ParentSeparator + invokedTypeFullName,
			StringComparison.Ordinal) ||
		invokedTypeFullName.EndsWith(Context.ParentSeparator + currentTypeName,
			StringComparison.Ordinal);

	private static BinaryMethod? FindCompiledMethod(BinaryExecutable binary, string currentTypeName,
		InvokeMethodInfo methodInfo)
	{
		if (binary.MethodsPerType.TryGetValue(currentTypeName, out var typeData) &&
			typeData.MethodGroups.TryGetValue(methodInfo.MethodName, out var overloads))
			return overloads.FirstOrDefault(candidate =>
				candidate.parameters.Count == methodInfo.ParameterNames.Length &&
				candidate.ReturnTypeName.EndsWith(methodInfo.ReturnTypeName, StringComparison.Ordinal));
		return null;
	}

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

	private static List<Instruction> RemapInstructions(IReadOnlyList<Instruction> instructions,
		BinaryMethod compiledMethod, Invoke invoke)
	{
		var returnRegister = ((ReturnInstruction)instructions[^1]).Register;
		var registerMap = new Dictionary<Register, Register> { [returnRegister] = invoke.Register };
		if (!TryMapParameterRegisters(instructions, compiledMethod, invoke, returnRegister,
			registerMap))
			return [];
		var nextRegister = ((int)invoke.Register + 1) % Enum.GetValues<Register>().Length;
		foreach (var register in instructions.SelectMany(GetRegisters))
			if (!registerMap.ContainsKey(register))
			{
				while (RegisterMapContainsValue(registerMap, (Register)nextRegister))
					nextRegister = (nextRegister + 1) % Enum.GetValues<Register>().Length;
				registerMap[register] = (Register)nextRegister;
				nextRegister = (nextRegister + 1) % Enum.GetValues<Register>().Length;
			}
		var remapped = new List<Instruction>(instructions.Count - 1);
		for (var index = 0; index < instructions.Count - 1; index++)
		{
			if (instructions[index] is LoadVariableToRegister loadVariable &&
				TryGetParameterIndex(compiledMethod, loadVariable.Identifier, out _))
				continue;
			remapped.Add(Clone(instructions[index], registerMap));
		}
		return remapped;
	}

	private static bool TryMapParameterRegisters(IReadOnlyList<Instruction> instructions,
		BinaryMethod compiledMethod, Invoke invoke, Register returnRegister,
		IDictionary<Register, Register> registerMap)
	{
		for (var index = 0; index < instructions.Count - 1; index++)
			switch (instructions[index])
			{
			case LoadVariableToRegister loadVariable:
				if (TryGetParameterIndex(compiledMethod, loadVariable.Identifier, out var parameterIndex))
				{
					var argumentRegister = invoke.MethodInfo.ArgumentRegisters[parameterIndex];
					if (loadVariable.Register == returnRegister && argumentRegister != invoke.Register)
						return false;
					registerMap[loadVariable.Register] = argumentRegister;
				}
				else if (UsesParameterAccessPath(compiledMethod, loadVariable.Identifier))
				{
					return false;
				}
				break;
			case ListCallInstruction listCall when UsesParameterAccessPath(compiledMethod,
				listCall.Identifier):
				return false;
			}
		return true;
	}

	private static bool TryGetParameterIndex(BinaryMethod compiledMethod, string identifier,
		out int parameterIndex)
	{
		for (parameterIndex = 0; parameterIndex < compiledMethod.parameters.Count; parameterIndex++)
			if (compiledMethod.parameters[parameterIndex].Name == identifier)
				return true;
		parameterIndex = -1;
		return false;
	}

	private static bool UsesParameterAccessPath(BinaryMethod compiledMethod, string identifier)
	{
		foreach (var parameter in compiledMethod.parameters)
			if (identifier.StartsWith(parameter.Name + ".", StringComparison.Ordinal) ||
				identifier.StartsWith(parameter.Name + "(", StringComparison.Ordinal))
				return true;
		return false;
	}

	private static bool RegisterMapContainsValue(IReadOnlyDictionary<Register, Register> registerMap,
		Register register) =>
		registerMap.Any(pair => pair.Value == register);

	private static IEnumerable<Register> GetRegisters(Instruction instruction) =>
		instruction switch
		{
			ListCallInstruction listCall => [listCall.Register, listCall.IndexValueRegister],
			BinaryInstruction binary => binary.Registers,
			RegisterInstruction registerInstruction => [registerInstruction.Register],
			_ => []
		};

	private static Instruction Clone(Instruction instruction,
		IReadOnlyDictionary<Register, Register> registerMap) =>
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
			Invoke nestedInvoke => new Invoke(registerMap[nestedInvoke.Register], new InvokeMethodInfo(
				nestedInvoke.MethodInfo.TypeFullName, nestedInvoke.MethodInfo.MethodName,
				nestedInvoke.MethodInfo.ParameterNames, nestedInvoke.MethodInfo.ReturnTypeName,
				nestedInvoke.MethodInfo.ArgumentRegisters.Select(register => registerMap[register]).
					ToArray(), nestedInvoke.MethodInfo.InstanceRegister.HasValue
					? registerMap[nestedInvoke.MethodInfo.InstanceRegister.Value]
					: null)),
			SetInstruction set => new SetInstruction(set.ValueInstance, registerMap[set.Register]),
			_ => instruction
		};
}