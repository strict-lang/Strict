using Strict.Language;

namespace Strict.Expressions;

/// <summary>
/// Constant, let, or mutable declaration in a method, often constant fixed value that is
/// optimized away. Very rarely should be mutable as this can't be optimized away so well, "let"
/// is fine as it is immutable after declaration.
/// </summary>
public class Declaration : ConcreteExpression
{
	public Declaration(Body scope, string name, Expression value, bool isMutable = false) :
		base(value.ReturnType, value.LineNumber, isMutable)
	{
		if (!name.IsWord())
			throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name);
		scope.AddVariable(name, value, isMutable);
		Name = name;
		Value = value;
	}

	public string Name { get; }
	public Expression Value { get; private set; }
	public override bool IsConstant => Value.IsConstant && !IsMutable;

	// ReSharper disable once NonReadonlyMemberInGetHashCode
	public override int GetHashCode() => Name.GetHashCode() ^ Value.GetHashCode();

	public override bool Equals(Expression? other) =>
		other is Declaration a && Equals(Name, a.Name) && Value.Equals(a.Value);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith(ConstantWithSpaceAtEnd, StringComparison.Ordinal)
			? TryParseDeclaration(body, line, ConstantWithSpaceAtEnd)
			: line.StartsWith(LetWithSpaceAtEnd, StringComparison.Ordinal)
				? TryParseDeclaration(body, line, LetWithSpaceAtEnd)
				: line.StartsWith(MutableWithSpaceAtEnd, StringComparison.Ordinal)
					? TryParseDeclaration(body, line, MutableWithSpaceAtEnd)
					: null;

	internal const string ConstantWithSpaceAtEnd = Keyword.Constant + " ";
	internal const string LetWithSpaceAtEnd = Keyword.Let + " ";
	internal const string MutableWithSpaceAtEnd = Keyword.Mutable + " ";

	public override string ToString() =>
		(IsConstant
			? ConstantWithSpaceAtEnd
			: IsMutable
				? MutableWithSpaceAtEnd
				: LetWithSpaceAtEnd) + Name + " = " + Value;

	/// <summary>
	/// Highly optimized parsing of assignments, skips over the mutable, grabs the name of the local
	/// variable, then skips over the space, equal and space characters and parses the rest, e.g.
	/// constant hello = "hello" + " " + "world"
	///					 ^ ^       ^ ^   ^ ^       END, using TryParseExpression with Range(12, 35)
	/// </summary>
	protected static Expression TryParseDeclaration(Body body, ReadOnlySpan<char> line,
		string declarationType)
	{
		var parts = line[declarationType.Length..].Split();
		parts.MoveNext();
		var name = parts.Current.ToString();
		if (!parts.MoveNext() || !parts.MoveNext())
			throw new MissingAssignmentValueExpression(body);
		var valueSpan = line[(declarationType.Length + name.Length + 1 + 1 + 1)..];
		if (declarationType == MutableWithSpaceAtEnd)
			return CreateMutableDeclaration(body, valueSpan, name);
		var value = body.Method.ParseExpression(body, valueSpan);
		var isActuallyConstant = IsExpressionFullyConstant(value);
		if (declarationType == LetWithSpaceAtEnd && isActuallyConstant)
			throw new LetUsesConstantValue(body, name, value);
		if (declarationType == ConstantWithSpaceAtEnd && !isActuallyConstant)
			throw new ConstantUsesNonConstantValue(body, name, value);
		return new Declaration(body, name, value);
	}

	private static Expression CreateMutableDeclaration(Body body, ReadOnlySpan<char> valueSpan,
		string name)
	{
		var value = valueSpan.IsFirstLetterUppercase() && (valueSpan.IsPlural() ||
			valueSpan.StartsWith(Base.List + '(' + Base.Mutable, StringComparison.Ordinal))
			? new List(body.Method.Type.GetType(valueSpan.ToString()))
			: body.Method.ParseExpression(body, valueSpan, true);
		return new Declaration(body, name, value, true);
	}

	private static bool IsExpressionFullyConstant(Expression expr)
	{
		if (!expr.IsConstant)
			return false;
		return expr switch
		{
			MethodCall methodCall => methodCall.Arguments.All(IsExpressionFullyConstant),
			MemberCall memberCall => IsExpressionFullyConstant(memberCall.Member.InitialValue!),
			List list => list.Values.All(IsExpressionFullyConstant),
			_ => true
		};
	}

	public sealed class ConstantUsesNonConstantValue(Body body, string name, Expression value)
		: ParsingFailed(body,
			$"Constant declaration uses non-constant value, use let instead: let {name} = {value}");

	public sealed class LetUsesConstantValue(Body body, string name, Expression value)
		: ParsingFailed(body,
			$"Let declaration uses only constant value, use constant instead: constant {name} = {value}");

	public sealed class MissingAssignmentValueExpression(Body body) : ParsingFailed(body);
	public void SetValue(Expression newValue) => Value = newValue;
}