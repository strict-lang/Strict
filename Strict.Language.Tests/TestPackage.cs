namespace Strict.Language.Tests
{
	/// <summary>
	/// Helper context to provide a bunch of helper types to make tests work.
	/// </summary>
	public class TestPackage : Package
	{
		public TestPackage() : base(nameof(TestPackage))
		{
			new Type(this, Base.Any, "method IsEqualTo(target Any) returns Boolean");
			new Type(this, Base.String, "method dummy");
			new Type(this, Base.Log, "method WriteLine(text String)");
			new Type(this, Base.Number, "method +(other Number) returns Number");
		}
	}
}