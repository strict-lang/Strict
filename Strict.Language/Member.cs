namespace Strict.Language;

public sealed class Member : NamedType
{
	public Member(Type definedIn, string nameAndType, Expression? value, bool usedMutableKeyword = false) : base(definedIn,
		nameAndType, value?.ReturnType)
	{
		Value = value;
		if (usedMutableKeyword)
			IsMutable = true;
		if (!Type.Name.StartsWith(Name.MakeFirstLetterUppercase(), StringComparison.Ordinal))
			CheckForNameWithDifferentTypeUsage(definedIn);
	}

	private void CheckForNameWithDifferentTypeUsage(Type definedIn)
	{
		try
		{
			var nameType = definedIn.GetType(Name.MakeFirstLetterUppercase());
			if (nameType != null && nameType != Type)
				throw new MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed(definedIn, Name, Type.Name);
		}
		catch (Context.TypeNotFound) { }
	}

	public Expression? Value { get; private set; }
	public bool IsPublic => char.IsUpper(Name[0]);
	public Expression[]? Constraints { get; private set; }

	public Member CloneWithImplementation(Type implementationType) =>
		new(Name, implementationType, IsMutable);

	private Member(string name, Type newType, bool isMutable) : base(newType, name, newType) =>
		IsMutable = isMutable;

	public void ParseConstraints(ExpressionParser parser, string[] constraintsText)
	{
		var expressions = new Expression[constraintsText.Length];
		for (var index = 0; index < constraintsText.Length; index++)
		{
			expressions[index] = parser.ParseExpression(
				new Body(new Method(Type, 0, parser, new[] { ConstraintsBody })), constraintsText[index]);
			if (expressions[index].ReturnType.Name != Base.Boolean)
				throw new InvalidConstraintExpression(Type, Name, constraintsText[index]);
		}
		Constraints = expressions;
	}

	public const string ConstraintsBody = nameof(ConstraintsBody);

	public sealed class InvalidConstraintExpression : ParsingFailed
	{
		public InvalidConstraintExpression(Type type, string memberName, string constraintText) : base(type, 0, $"Constraint: {constraintText} Member: {memberName}") { }
	}

	public sealed class MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed : ParsingFailed
	{
		public MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed(Type type, string nameType,
			string typeName) : base(type, 0, $"Name {nameType} and type {typeName} are not matching") { }
	}

	public void UpdateValue(Expression newExpression, Body bodyForErrorMessage)
	{
		if (!IsMutable && Value != null)
			throw new Body.ValueIsNotMutableAndCannotBeChanged(bodyForErrorMessage, Name);
		Value = newExpression;
	}
}