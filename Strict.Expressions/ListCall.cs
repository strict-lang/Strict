using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ListCall(Expression list, Expression index, Expression? secondIndex = null,
	Expression? originalIndex = null) : ConcreteExpression(
	list.ReturnType is GenericTypeImplementation listReturnType
		? listReturnType.ImplementationTypes[0]
		: list.ReturnType, list.LineNumber, list.IsMutable)
{
	public Expression List { get; } = list;
	public Expression Index { get; } = index;
	public Expression? SecondIndex { get; } = secondIndex;
	public Expression OriginalIndex { get; } = originalIndex ?? index;

	public static Expression? TryParse(Body body, Expression? variable,
		IReadOnlyList<Expression> arguments)
	{
		if (variable is null or MethodCall || arguments.Count == 0)
			return variable;
		if (!variable.ReturnType.IsIterator)
			return variable is VariableCall { Variable.Name: Type.OuterLowercase }
				? new ListCall(variable, arguments[0])
				: variable.ReturnType.IsError
					? MethodCall.CreateFromMethodCall(body, variable.ReturnType, arguments, variable)
					: body.IsFakeBodyForMemberInitialization && arguments.Count == 1
						? arguments[0]
						: throw new MethodExpressionParser.InvalidArgumentItIsNotMethodOrListCall(body,
							variable, arguments);
		if (arguments.Count > 2)
			throw new OnlyOneOrTwoArgumentsAreSupported(body, variable, arguments.Count);
		var flattenedIndex = arguments.Count == 1
			? arguments[0]
			: CreateFlattenedIndex(body, variable, arguments[0], arguments[1]);
		var listCall = CreateListCallAndCheckIndexBounds(body, variable, flattenedIndex);
		return arguments.Count == 1
			? listCall
			: new ListCall(listCall.List, listCall.Index, arguments[1], arguments[0]);
	}

	private static Expression CreateFlattenedIndex(Body body, Expression listVariable,
		Expression xIndex, Expression yIndex)
	{
		var sizeWidth = GetSizeWidthAccess(body, listVariable);
		var multipliedHeight = new Binary(sizeWidth,
			sizeWidth.ReturnType.GetMethod(BinaryOperator.Multiply, [yIndex]), [yIndex]);
		return new Binary(xIndex, xIndex.ReturnType.GetMethod(BinaryOperator.Plus,
			[multipliedHeight]), [multipliedHeight]);
	}

	private static Expression GetSizeWidthAccess(Body body, Expression listVariable)
	{
		if (listVariable is not MemberCall memberCall)
			throw new TwoDimensionalListAccessRequiresSizeMember(body, listVariable);
		var sizeMember = memberCall.Member.DefinedIn.FindMember("Size");
		if (sizeMember == null)
			throw new TwoDimensionalListAccessRequiresSizeMember(body, listVariable);
		var widthMember = sizeMember.Type.FindMember("Width");
		if (widthMember == null)
			throw new TwoDimensionalListAccessRequiresSizeMember(body, listVariable);
		var sizeCall = new MemberCall(memberCall.Instance, sizeMember, body.CurrentFileLineNumber);
		return new MemberCall(sizeCall, widthMember, body.CurrentFileLineNumber);
	}

	private static ListCall CreateListCallAndCheckIndexBounds(Body body, Expression listVariable,
		Expression index)
	{
		var indexValue = GetIndexValue(index);
		if (indexValue == null)
			return new ListCall(listVariable, index);
		List? specifiedList = null;
		if (listVariable is MemberCall memberCall)
			specifiedList = CheckConstraints(body, listVariable, memberCall, (int)indexValue);
		else if (listVariable is VariableCall variableCall)
			specifiedList = variableCall.Variable.InitialValue as List;
		if (specifiedList is { Values.Count: > 0 })
		{
			var normalizedIndex = NormalizeIndex((int)indexValue, specifiedList.Values.Count);
			if (normalizedIndex < 0 || normalizedIndex >= specifiedList.Values.Count)
				throw new IndexAboveConstantListLength(body, (int)indexValue, specifiedList);
		}
		return new ListCall(listVariable, index);
	}

	private static int NormalizeIndex(int indexValue, int length) =>
		indexValue < 0
			? length + indexValue
			: indexValue;

	private static List? CheckConstraints(Body body, Expression listVariable, MemberCall memberCall,
		int indexValue)
	{
		var lengthConstraint = FindLengthConstraint(memberCall.Member.Constraints);
		var constrainedLength = lengthConstraint != null
			? ExtractLengthValue(lengthConstraint)
			: null;
		if (constrainedLength.HasValue)
		{
			var normalizedIndex = NormalizeIndex(indexValue, constrainedLength.Value);
			if (normalizedIndex < 0 || normalizedIndex >= constrainedLength.Value)
				throw new IndexViolatesListConstraint(body, indexValue, listVariable, lengthConstraint!);
		}
		return memberCall.Instance as List;
	}

	private static int? GetIndexValue(Expression index)
	{
		if (index is VariableCall { Variable.IsMutable: false } variableCall)
			index = variableCall.Variable.InitialValue;
		if (index is MemberCall
			{
				IsMutable: false, Member: { InitialValue: not null, IsConstant: true }
			} memberCall)
			index = memberCall.Member.InitialValue;
		if (index is Number indexNumber)
			return (int)indexNumber.Data.number;
		return null;
	}

	private static Expression? FindLengthConstraint(IReadOnlyList<Expression>? constraints)
	{
		if (constraints != null)
			foreach (var constraint in constraints)
				if (constraint is Binary { Method.Name: BinaryOperator.Is } binary &&
					binary.Instance?.ToString() == "Length")
					return binary;
		return null;
	}

	private static int? ExtractLengthValue(Expression lengthConstraint) =>
		lengthConstraint is Binary { Arguments.Count: > 0 } binary &&
		binary.Arguments[0] is Number lengthNumber
			? (int)lengthNumber.Data.number
			: null;

	public sealed class OnlyOneOrTwoArgumentsAreSupported(Body body, Expression list,
		int argumentCount) : ParsingFailed(body, list + " only supports 1 or 2 arguments, got " +
		argumentCount, list.ReturnType);

	public sealed class TwoDimensionalListAccessRequiresSizeMember(Body body, Expression list)
		: ParsingFailed(body, list + " needs a sibling Size member for 2D indexing", list.ReturnType);

	public sealed class IndexAboveConstantListLength(Body body, int index, List list) : ParsingFailed(
		body, $"Index {index} is out of range for list {list} with length {list.Values.Count}. " +
		$"Valid indices are 0 to {list.Values.Count - 1} and negative indices down to {-list.Values.Count}.", list.ReturnType);

	public sealed class IndexViolatesListConstraint(Body body, int index, Expression list,
		Expression constraint) : ParsingFailed(body, $"Index {index} is not allowed based on the " +
		$"constraint on the {list} definition: {constraint}.", list.ReturnType);

	public override bool IsConstant => List.IsConstant && Index.IsConstant;

	public override string ToString() =>
		SecondIndex == null
			? $"{List}({Index})"
			: $"{List}({OriginalIndex}, {SecondIndex})";

	//ncrunch: no coverage start
	public override bool Equals(Expression? other) =>
		ReferenceEquals(this, other) ||
		other is ListCall listCall && List.Equals(listCall.List) && Index.Equals(listCall.Index) &&
		OriginalIndex.Equals(listCall.OriginalIndex) && Equals(SecondIndex, listCall.SecondIndex);

	public override int GetHashCode() =>
		List.GetHashCode() ^ Index.GetHashCode() ^ OriginalIndex.GetHashCode() ^
		(SecondIndex?.GetHashCode() ?? 0);
}