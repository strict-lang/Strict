namespace Strict.Language.Tests
{
	/// <summary>
	/// Helper context to provide a bunch of helper types to make tests work.
	/// </summary>
	public class TestPackage : Package
	{
		public TestPackage() : base(nameof(TestPackage))
		{
			new Type(this, Base.Any, "is(any) returns Boolean");
			new Type(this, Base.Text, "dummy");
			new Type(this, Base.Log, "Write(text)");
			new Type(this, Base.Number, "+(other) returns Number");
		}
	}
}