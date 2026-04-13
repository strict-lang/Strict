using Strict.Expressions;
using Strict.Language;

namespace Strict.HighLevelRuntime;

/// <summary>
/// Keeps track of things happening while executing Interpreter, used together with TestRunner.
/// </summary>
//ncrunch: no coverage start
public sealed record Statistics
{
	public int PackagesTested => packagesTested;
	private int packagesTested;
	public void IncrementPackagesTested() => Interlocked.Increment(ref packagesTested);
	public int TypesTested => typesTested;
	private int typesTested;
	public void IncrementTypesTested() => Interlocked.Increment(ref typesTested);
	public int MethodsTested => methodsTested;
	private int methodsTested;
	public void IncrementMethodsTested() => Interlocked.Increment(ref methodsTested);
	public int TestExpressions { get; internal set; }
	public int MethodCount { get; internal set; }
	public int ExpressionCount { get; internal set; }
	public int BodyCount { get; internal set; }
	public int IfCount { get; internal set; }
	public int SelectorIfCount { get; internal set; }
	public int ForCount { get; internal set; }
	public int MethodCallCount { get; internal set; }
	public int MemberCallCount { get; internal set; }
	public int ListCallCount { get; internal set; }
	public int BinaryCount { get; internal set; }
	public int ArithmeticCount { get; internal set; }
	public int CompareCount { get; internal set; }
	public int LogicalOperationCount { get; internal set; }
	public int UnaryCount { get; internal set; }
	public int FromCreationsCount { get; internal set; }
	public int ToConversionCount { get; internal set; }
	public int ReturnCount { get; internal set; }
	public int VariableDeclarationCount { get; internal set; }
	public int VariableCallCount { get; internal set; }
	public int MutableDeclarationCount { get; internal set; }
	public int MutableUsageCount { get; internal set; }
	public int FindVariableCount { get; internal set; }
	public int FindTypeCount => Context.FindTypeCount;
	public int ValueInstanceCreatedCount => ValueInstance.CreatedCount;
	public int ValueInstanceComplexEqualsCalls => ValueInstance.ComplexEqualsCalls;

	public void Reset()
	{
		packagesTested = 0;
		typesTested = 0;
		methodsTested = 0;
		TestExpressions = 0;
		MethodCount = 0;
		ExpressionCount = 0;
		BodyCount = 0;
		IfCount = 0;
		SelectorIfCount = 0;
		ForCount = 0;
		MethodCallCount = 0;
		MemberCallCount = 0;
		ListCallCount = 0;
		BinaryCount = 0;
		ArithmeticCount = 0;
		CompareCount = 0;
		LogicalOperationCount = 0;
		UnaryCount = 0;
		FromCreationsCount = 0;
		ToConversionCount = 0;
		ReturnCount = 0;
		VariableDeclarationCount = 0;
		VariableCallCount = 0;
		MutableDeclarationCount = 0;
		MutableUsageCount = 0;
		FindVariableCount = 0;
		Context.FindTypeCount = 0;
		ValueInstance.ComplexEqualsCalls = 0;
	}
}