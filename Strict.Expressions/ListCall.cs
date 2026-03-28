using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ListCall(Expression list, Expression index) : ConcreteExpression(
	list.ReturnType is GenericTypeImplementation listReturnType
		? listReturnType.ImplementationTypes[0]
		: list.ReturnType, list.LineNumber, list.IsMutable)
{
	public Expression List { get; } = list;
	public Expression Index { get; } = index;

	public static Expression? TryParse(Body body, Expression? variable,
		IReadOnlyList<Expression> arguments) =>
		variable is not null and not MethodCall && arguments.Count > 0
			? variable.ReturnType.IsIterator
				? CreateListCallAndCheckIndexBounds(body, variable, arguments[0])
				: variable is VariableCall { Variable.Name: Type.OuterLowercase }
					? new ListCall(variable, arguments[0])
					: variable.ReturnType.IsError
						? MethodCall.CreateFromMethodCall(body, variable.ReturnType, arguments, variable)
						: body.IsFakeBodyForMemberInitialization && arguments.Count == 1
							? arguments[0]
							: throw new MethodExpressionParser.InvalidArgumentItIsNotMethodOrListCall(body,
								variable, arguments)
			: variable;

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

	public sealed class IndexAboveConstantListLength(Body body, int index, List list) : ParsingFailed(
		body, $"Index {index} is out of range for list {list} with length {list.Values.Count}. " +
		$"Valid indices are 0 to {list.Values.Count - 1} and negative indices down to {-list.Values.Count}.", list.ReturnType);

	public sealed class IndexViolatesListConstraint(Body body, int index, Expression list,
		Expression constraint) : ParsingFailed(body, $"Index {index} is not allowed based on the " +
		$"constraint on the {list} definition: {constraint}.", list.ReturnType);

	public override bool IsConstant => List.IsConstant && Index.IsConstant;
	public override string ToString() => $"{List}({Index})";

	//ncrunch: no coverage start
	public override bool Equals(Expression? other) =>
		ReferenceEquals(this, other) ||
		other is ListCall lc && List.Equals(lc.List) && Index.Equals(lc.Index);

	public override int GetHashCode() => List.GetHashCode() ^ Index.GetHashCode();
}