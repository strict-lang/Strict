using Strict.Expressions;
using Strict.Runtime.Statements;

namespace Strict.Optimizers.Tests;

public class TestOptimizers
{
	public ValueInstance Num(double value) => new(numberType, value);
	protected readonly Type numberType = TestPackage.Instance.GetType(Type.Number);

	protected List<Statement> Optimize(InstructionOptimizer optimizer, List<Statement> statements,
		int expectedCount)
	{
		var optimizedStatements = optimizer.Optimize(statements);
		Assert.That(optimizedStatements, Has.Count.EqualTo(expectedCount));
		return optimizedStatements;
	}
}