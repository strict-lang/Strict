namespace Strict.Language.Expressions
{
	/// <summary>
	/// Any expression with a fixed value, often optimized from all known code trees, often parameters
	/// like the derived classes <see cref="Number"/>, <see cref="Boolean"/> or <see cref="Text"/>.
	/// </summary>
	public class Value : Expression
	{
		public Value(Type valueType, object data) : base(valueType) => Data = data;
		public object Data { get; }//ncrunch: no coverage
		public override string ToString() => Data.ToString()!;
	}
}