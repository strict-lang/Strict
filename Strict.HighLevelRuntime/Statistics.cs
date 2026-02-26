using Strict.Language;

namespace Strict.HighLevelRuntime;

/// <summary>
/// Keeps track of things happening the classes here, usually used together with TestRunner.
/// </summary>
public sealed record Statistics
{
	public int MethodCount { get; set; }
	public int MethodTested { get; set; }
	public int TestsCount { get; set; }
	public int ExpressionCount { get; set; }

	//TODO: these are not longer tracked, either remove or fix
	public int ValueInstanceCount { get; set; }
	public int BooleanCount { get; set; }
	public int NumberCount { get; set; }
	public int TextCount { get; set; }
	public int ListCount { get; set; }
	public int DictionaryCount { get; set; }

	public int BodyCount { get; set; }
	public int IfCount { get; set; }
	public int SelectorIfCount { get; set; }
	public int ForCount { get; set; }
	public int MethodCallCount { get; set; }
	public int MemberCallCount { get; set; }
	public int ListCallCount { get; set; }
	public int BinaryCount { get; set; }
	public int ArithmeticCount { get; set; }
	public int CompareCount { get; set; }
	public int LogicalOperationCount { get; set; }
	public int UnaryCount { get; set; }
	public int FromCreationsCount { get; set; }
	public int ToConversionCount { get; set; }
	public int ReturnCount { get; set; }
	public int VariableDeclarationCount { get; set; }
	public int VariableCallCount { get; set; }
	public int MutableDeclarationCount { get; set; }
	public int MutableUsageCount { get; set; }
	public int FindTypeCount => Context.FindTypeCount;
	public int FindVariableCount { get; set; }
}

