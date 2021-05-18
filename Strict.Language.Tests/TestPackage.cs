namespace Strict.Language.Tests
{
	/// <summary>
	/// Helper context to provide a bunch of helper types to make tests work.
	/// </summary>
	public class TestPackage : Package
	{
		public TestPackage() : base(nameof(TestPackage))
		{
			new Type(this, Base.Any, null).Parse("is(any) returns Boolean");
			new Type(this, Base.Text, null).Parse("dummy");
			new Type(this, Base.Log, null).Parse("WriteLine(text)");
			new Type(this, Base.Number, null).Parse("+(other) returns Number");
		}
	}
}