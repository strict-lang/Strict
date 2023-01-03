﻿namespace Strict.Language;

public sealed class Member : NamedType
{
	public Member(Type definedIn, string nameAndType, Expression? value, bool usedMutableKeyword = false) : base(definedIn,
		nameAndType, value?.ReturnType)
	{
		Value = value;
		if (usedMutableKeyword)
			IsMutable = true;
	}

	public Expression? Value { get; set; }
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
				new Body(new Method(Type, 0, parser, new[] { "EmptyBody" })), constraintsText[index]);
			if (expressions[index].ReturnType.Name != Base.Boolean)
				throw new InvalidConstraintExpression(Type, Name, constraintsText[index]);
		}
		Constraints = expressions;
	}

	public sealed class InvalidConstraintExpression : ParsingFailed
	{
		public InvalidConstraintExpression(Type type, string memberName, string constraintText) : base(type, 0, $"Constraint: {constraintText} Member: {memberName}") { }
	}
}