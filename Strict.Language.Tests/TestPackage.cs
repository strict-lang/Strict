namespace Strict.Language.Tests;

/// <summary>
/// Helper context to provide a bunch of helper types to make tests work.
/// </summary>
public class TestPackage : Package
{
	public TestPackage() : base(nameof(TestPackage))
	{
		new Type(this, Base.Any, null!).Parse("is(other) returns Boolean");
		new Type(this, Base.Number, null!).Parse(@"+(other) returns Number
	return self + other
*(other) returns Number
	return self * other
-(other) returns Number
	return self - other
/(other) returns Number
	return self / other");
		new Type(this, Base.Count, null!).Parse(@"implement Number
from(number)
	value = number
Increase
	Count(5).Increase is 6
	self = self + 1");
		new Type(this, Base.Character, null!).Parse(@"implement Number
from(number)
	value = number");
		new Type(this, Base.Text, null!).Parse(@"has Characters
Run
	value is not """"
+(other) returns Text
	return value");
		new Type(this, Base.Log, null!).Parse(@"has Text
Write(text)
	return Text
Write(number)
	return number");
		new Type(this, Base.List, null!).Parse(@"+(other) returns List
-(other) returns List
is(other) returns Boolean
*(other) returns List");
		new Type(this, Base.Boolean, null!).Parse(@"not returns Boolean
is(any) returns Boolean");
	}
}