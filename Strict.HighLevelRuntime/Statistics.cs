using Strict.Language;

namespace Strict.HighLevelRuntime;

/// <summary>
/// Keeps track of things happening the classes here, usually used together with TestRunner.
/// </summary>
public sealed record Statistics
{
	public int MethodCount { get; internal set; }
	public int MethodTested { get; internal set; }
	public int TestsCount { get; internal set; }
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
	public int FindTypeCount => Context.FindTypeCount;
	public int FindVariableCount { get; internal set; }
}

