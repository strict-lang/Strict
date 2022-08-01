namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public sealed class MemberCall : Expression
{
	public MemberCall(Expression? instance, Member member) : base(member.Type)
	{
		Instance = instance;
		Member = member;
	}

	public Expression? Instance { get; }
	public Member Member { get; }

	public override string ToString() =>
		Instance != null
			? $"{Instance}.{Member.Name}"
			: Member.Name;
}

// ReSharper disable once HollowTypeName
public sealed class VariableCall : Expression
{
	public VariableCall(string name, Expression value) : base(value.ReturnType)
	{
		Name = name;
		Value = value;
	}

	public string Name { get; }
	public Expression Value { get; }
	public override string ToString() => Name;
}

// ReSharper disable once HollowTypeName
public sealed class ParameterCall : Expression
{
	public ParameterCall(Parameter parameter) : base(parameter.Type) => Parameter = parameter;
	public Parameter Parameter { get; }
	public override string ToString() => Parameter.Name;
}