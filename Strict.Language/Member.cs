namespace Strict.Language;

public sealed class Member : NamedType
{
	public Member(Type definedIn, string nameAndType, Type? initialValueType, int lineNumber = 0,
		string usedKeyword = Keyword.Has) : base(definedIn, nameAndType, initialValueType)
	{
		DefinedIn = definedIn;
		LineNumber = lineNumber;
		if (usedKeyword == Keyword.Mutable)
			IsMutable = true;
		else if (usedKeyword == Keyword.Constant)
			IsConstant = true;
		if (Name != Type.ValueLowercase &&
			!Name.StartsWith("Is", StringComparison.OrdinalIgnoreCase) &&
			!Type.Name.StartsWith(Name.MakeFirstLetterUppercase(), StringComparison.Ordinal))
			CheckForNameWithDifferentTypeUsage(definedIn);
	}

	private void CheckForNameWithDifferentTypeUsage(Type definedIn)
	{
		var nameType = definedIn.Package.FindType(Name.MakeFirstLetterUppercase());
		if (nameType != null && nameType != Type)
			throw new MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed(definedIn, Name, Type.Name);
	}

	public Type DefinedIn { get; }
	public Expression? InitialValue { get; internal set; }
	internal string? InitialValueText { get; set; }
	public int LineNumber { get; }
	public bool IsPublic => char.IsUpper(Name[0]);
	public Expression[]? Constraints { get; private set; }

	public Member CloneWithImplementation(Type implementationType) =>
		new(Name, implementationType, IsMutable, IsConstant);

	private Member(string name, Type newType, bool isMutable, bool isConstant)
		: base(newType, name, newType)
	{
		DefinedIn = newType;
		IsMutable = isMutable;
		IsConstant = isConstant;
	}

	public void ParseConstraints(ExpressionParser parser, string[] constraintsText)
	{
		var expressions = new Expression[constraintsText.Length];
		var body = new Body(new Method(Type, 0, parser, [ConstraintsBody]));
		AddContainingTypeMembersAsConstraintVariables(body);
		for (var index = 0; index < constraintsText.Length; index++)
		{
			expressions[index] = parser.ParseExpression(body, constraintsText[index]);
			if (!expressions[index].ReturnType.IsBoolean)
				throw new InvalidConstraintExpression(Type, Name, constraintsText[index]);
		}
		Constraints = expressions;
	}

	private void AddContainingTypeMembersAsConstraintVariables(Body body)
	{
		foreach (var member in DefinedIn.Members)
			if (Type.FindMember(member.Name) == null)
				body.AddVariable(member.Name, new ConstraintMemberReference(member.Name, member.Type), false);
	}

	private sealed class ConstraintMemberReference(string memberName, Type returnType)
		: Expression(returnType)
	{
		public override bool IsConstant => false;
		public override string ToString() => memberName;

		public override bool Equals(Expression? other) =>
			ReferenceEquals(this, other) || other is ConstraintMemberReference reference &&
			reference.ReturnType == ReturnType && reference.ToString() == memberName;

		public override int GetHashCode() => HashCode.Combine(memberName, ReturnType);
	}

	public const string ConstraintsBody = nameof(ConstraintsBody);

	public sealed class	InvalidConstraintExpression(Type type, string memberName,
		string constraintText) : ParsingFailed(type, 0,
		$"Constraint: {constraintText} Member: {memberName}");

	public sealed class MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed(Type type,
		string nameType, string typeName)
		: ParsingFailed(type, 0, $"Name {nameType} and type {typeName} are not matching, it is not " +
			$"allowed to use reserved types as member names if the type is completely different!");

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