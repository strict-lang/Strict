using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Optimizers;

/// <summary>
/// Removes stores to variables that are never loaded again. A StoreVariableInstruction is dead
/// if the variable identifier never appears in any LoadVariableToRegister,
/// StoreFromRegisterInstruction, or as the instance of an Invoke instruction.
/// Member variables are always kept as they may be accessed externally.
/// </summary>
public sealed class DeadStoreEliminator : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		var usedVariables = CollectUsedVariables(instructions);
		instructions.RemoveAll(instruction =>
			instruction is StoreVariableInstruction { IsMember: false } store &&
			!usedVariables.Contains(store.Identifier));
		return instructions;
	}

	private static HashSet<string> CollectUsedVariables(List<Instruction> instructions)
	{
		var used = new HashSet<string>();
		foreach (var instruction in instructions)
			switch (instruction.InstructionType)
			{
			case InstructionType.LoadVariableToRegister:
				used.Add(((LoadVariableToRegister)instruction).Identifier);
				break;
			case InstructionType.StoreRegisterToVariable:
				used.Add(((StoreFromRegisterInstruction)instruction).Identifier);
				break;
			case InstructionType.Invoke:
				CollectVariableCallsFromInvoke((Invoke)instruction, used);
				break;
			}
		return used;
	}

	private static void CollectVariableCallsFromInvoke(Invoke invoke, HashSet<string> used)
	{
		if (invoke.Method.Instance is VariableCall instanceVar)
			used.Add(instanceVar.Variable.Name);
		foreach (var argument in invoke.Method.Arguments)
			CollectVariableCallsFromExpression(argument, used);
	}

	private static void CollectVariableCallsFromExpression(Expression expression,
		HashSet<string> used)
	{
		if (expression is VariableCall variableCall)
			used.Add(variableCall.Variable.Name);
		else if (expression is MethodCall methodCall)
		{
			if (methodCall.Instance != null)
				CollectVariableCallsFromExpression(methodCall.Instance, used);
			foreach (var arg in methodCall.Arguments)
				CollectVariableCallsFromExpression(arg, used);
		}
	}
}