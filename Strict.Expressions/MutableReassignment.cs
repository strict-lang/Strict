using Strict.Language;

namespace Strict.Expressions;

public sealed class MutableReassignment : ConcreteExpression
{
	private MutableReassignment(Body scope, Expression target, Expression newValue) : base(
		newValue.ReturnType, newValue.LineNumber, true)
	{
		if (target is { IsMutable: false } && scope.Method.Name != Method.From)
			throw new Body.ValueIsNotMutableAndCannotBeChanged(scope, target.ToString());
		Name = target switch
		{
			VariableCall variableCall => variableCall.Variable.Name,
			ParameterCall parameterCall => parameterCall.Parameter.Name,
			MemberCall memberCall => memberCall.Member.Name,
			_ => target.ToString()
		};
		Target = target;
		Value = newValue;
		var contextType = (target as MemberCall)?.Instance?.ReturnType ?? scope.Method.Type;
		if (target is not ListCall)
			scope.CheckIfWeCouldUpdateMutableParameterOrVariable(contextType, Name, Value);
	}

	public string Name { get; }
	public Expression Target { get; }
	public Expression Value { get; }

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		ContainsAssignmentOutsideString(line)
			? TryParseReassignment(body, line)
			: null;

	private static bool ContainsAssignmentOutsideString(ReadOnlySpan<char> input)
	{
		var inText = false;
		for (var i = 0; i < input.Length - 2; i++)
		{
			if (input[i] == '"')
				inText = !inText;
			if (!inText && input[i] == ' ' && input[i + 1] == '=' && input[i + 2] == ' ')
				return true;
		}
		return false;
	}

	private static Expression TryParseReassignment(Body body, ReadOnlySpan<char> line)
	{
		var parts = line.Split('=', StringSplitOptions.TrimEntries);
		parts.MoveNext();
		var expression = body.Method.ParseExpression(body, parts.Current, true);
		var newExpression = body.Method.ParseExpression(body, line[(parts.Current.Length + 3)..]);
		if (!newExpression.ReturnType.IsSameOrCanBeUsedAs(expression.ReturnType, false))
			throw new ValueTypeNotMatchingWithAssignmentType(body, expression.ReturnType,
				newExpression.ReturnType);
		return new MutableReassignment(body, expression, newExpression);
	}

	public sealed class ValueTypeNotMatchingWithAssignmentType(Body body,
		Language.Type currentValueType, Language.Type newValueType) : ParsingFailed(body,
		$"Cannot assign {newValueType} value type to {currentValueType} (Package="+newValueType.Package+" == Package="+currentValueType.Package+": "+(newValueType.Package==currentValueType.Package)+") member or variable");

	public override bool IsConstant => false;
	public override string ToString() => Name + " = " + Value;

	//ncrunch: no coverage start
	public override bool Equals(Expression? other) =>
		ReferenceEquals(this, other) || other is MutableReassignment mr && Name == mr.Name &&
		Target.Equals(mr.Target) && Value.Equals(mr.Value);

	public override int GetHashCode() => Name.GetHashCode() ^ Value.GetHashCode();
}