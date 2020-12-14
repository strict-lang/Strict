namespace Strict.Language
{
	public class Member : NamedType
	{
		public Member(string name, Expression value) : base(name, value.ReturnType) =>
			Value = value;

		public Expression Value { get; init; }

		public Member(Type definedIn, string nameAndType, Expression value) : base(definedIn,
			nameAndType) =>
			Value = value;
	}
}