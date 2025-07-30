using Strict.Language;

namespace Strict.Expressions;

public sealed class ListCall(Expression list, Expression index) : ConcreteExpression(
	list.ReturnType is GenericTypeImplementation listReturnType
		? listReturnType.ImplementationTypes[0]
		: list.ReturnType, list.IsMutable)
{
	public Expression List { get; } = list;
	public Expression Index { get; } = index;

	public static Expression? TryParse(Body body, Expression? variable,
		IReadOnlyList<Expression> arguments) =>
		variable is null or MethodCall
			? variable
			: arguments.Count > 0
				? variable.ReturnType.IsIterator
					? CreateListCallAndCheckIndexBounds(body, variable, arguments[0])
					: throw new MethodExpressionParser.InvalidArgumentItIsNotMethodOrListCall(body, variable,
						arguments)
				: variable;

	private static ListCall CreateListCallAndCheckIndexBounds(Body body, Expression listVariable,
		Expression index)
	{
		var indexValue = GetIndexValue(index);
		if (indexValue == null)
			return new ListCall(listVariable, index);
		if (indexValue < 0)
			throw new NegativeIndexIsNeverAllowed(body, listVariable);
		List? specifiedList = null;
		if (listVariable is MemberCall memberCall)
			specifiedList = CheckConstraints(body, listVariable, memberCall, (int)indexValue);
		else if (listVariable is VariableCall variableCall)
			specifiedList = variableCall.Variable.InitialValue as List;
		if (specifiedList is { Values.Count: > 0 } && indexValue >= specifiedList.Values.Count)
			throw new IndexAboveConstantListLength(body, (int)indexValue, specifiedList);
		return new ListCall(listVariable, index);
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
			return (int)(double)indexNumber.Data;
		return null;
	}

	private static List? CheckConstraints(Body body, Expression listVariable, MemberCall memberCall,
		int indexValue)
	{
		var lengthConstraint = FindLengthConstraint(memberCall.Member.Constraints);
		if (lengthConstraint != null && indexValue >= ExtractLengthValue(lengthConstraint))
			throw new IndexViolatesListConstraint(body, indexValue, listVariable, lengthConstraint);
		return memberCall.Instance as List;
	}

	private static Expression? FindLengthConstraint(IEnumerable<Expression>? constraints)
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
			? (int)(double)lengthNumber.Data
			: null;

	public sealed class NegativeIndexIsNeverAllowed(Body body, Expression list) :
		ParsingFailed(body, $"Negative index is not allowed for {list}.", list.ReturnType);

	public sealed class IndexAboveConstantListLength(Body body, int index, List list) : ParsingFailed(
		body, $"Index {index} is out of range for list {list} with length {list.Values.Count}. " +
		$"Valid indices are 0 to {list.Values.Count - 1}.", list.ReturnType);

	public sealed class IndexViolatesListConstraint(Body body, int index, Expression list,
		Expression constraint) : ParsingFailed(body, $"Index {index} is not allowed based on the " +
		$"constraint on the {list} definition: {constraint}.", list.ReturnType);

	public override string ToString() => $"{List}({Index})";
}