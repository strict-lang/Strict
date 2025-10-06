namespace Strict.Language;

public sealed class Member : NamedType
{
	public Member(Type definedIn, string nameAndType, Expression? initialValue,
		string usedKeyword = Keyword.Has) : base(definedIn, nameAndType, initialValue?.ReturnType)
	{
		InitialValue = initialValue;
		if (usedKeyword == Keyword.Mutable)
			IsMutable = true;
		else if (usedKeyword == Keyword.Constant)
			IsConstant = true;
		if (!Type.Name.StartsWith(Name.MakeFirstLetterUppercase(), StringComparison.Ordinal))
			CheckForNameWithDifferentTypeUsage(definedIn);
	}

	private void CheckForNameWithDifferentTypeUsage(Type definedIn)
	{
		var nameType = definedIn.TryGetType(Name.MakeFirstLetterUppercase());
		if (nameType != null && nameType != Type)
			throw new MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed(definedIn, Name, Type.Name);
	}

	public Expression? InitialValue { get; internal set; }
	public bool IsPublic => char.IsUpper(Name[0]);
	public Expression[]? Constraints { get; private set; }

	public Member CloneWithImplementation(Type implementationType) =>
		new(Name, implementationType, IsMutable, IsConstant);

	private Member(string name, Type newType, bool isMutable, bool isConstant) : base(newType, name,
		newType)
	{
		IsMutable = isMutable;
		IsConstant = isConstant;
	}

	public void ParseConstraints(ExpressionParser parser, string[] constraintsText)
	{
		var expressions = new Expression[constraintsText.Length];
		for (var index = 0; index < constraintsText.Length; index++)
		{
			expressions[index] = parser.ParseExpression(
				new Body(new Method(Type, 0, parser, [ConstraintsBody])), constraintsText[index]);
			if (expressions[index].ReturnType.Name != Base.Boolean)
				throw new InvalidConstraintExpression(Type, Name, constraintsText[index]);
		}
		Constraints = expressions;
	}

	public const string ConstraintsBody = nameof(ConstraintsBody);

	public sealed class	InvalidConstraintExpression(Type type, string memberName,
		string constraintText) : ParsingFailed(type, 0,
		$"Constraint: {constraintText} Member: {memberName}");

	public sealed class MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed(Type type,
		string nameType, string typeName)
		: ParsingFailed(type, 0, $"Name {nameType} and type {typeName} are not matching");

	public void CheckIfWeCouldUpdateValue(Expression newExpression, Body bodyForErrorMessage)
	{
		if (!IsMutable)
			throw new Body.ValueIsNotMutableAndCannotBeChanged(bodyForErrorMessage, Name);
		if (!newExpression.ReturnType.IsSameOrCanBeUsedAs(Type))
			throw new NewExpressionDoesNotMatchMemberType(bodyForErrorMessage, newExpression, this);
	}

	public class NewExpressionDoesNotMatchMemberType(Body body, Expression newExpression,
		Member member) : ParsingFailed(body, newExpression.ToStringWithType() +
		" cannot be assigned to " + member, member.Type);

	public override string ToString() =>
		(IsMutable
			? Type.MutableWithSpaceAtEnd
			: IsConstant
				? Type.ConstantWithSpaceAtEnd
				: "") + base.ToString();
}