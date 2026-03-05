using Strict.Expressions;

namespace Strict.Runtime.Statements;

public abstract class InstanceStatement(Instruction instruction, ValueInstance valueInstance) : Statement(instruction)
{
	public ValueInstance ValueInstance { get; } = valueInstance;
	public override string ToString() => $"{Instruction} {ValueInstance.ToExpressionCodeString()}";
}