namespace Strict.Language.Tests;

/// <summary>
/// Helper context to provide a bunch of helper types to make tests work.
/// </summary>
public class TestPackage : Package
{
	public TestPackage() : base(nameof(TestPackage))
	{
		new Type(this, new FileData(Base.Any, new[] { "is(other) returns Boolean" }), null!);
		new Type(this, new FileData(Base.Number, @"+(other) returns Number
	return self + other
*(other) returns Number
	return self * other
-(other) returns Number
	return self - other
/(other) returns Number
	return self / other".SplitLines()), null!);
		new Type(this, new FileData(Base.Count, @"implement Number
from(number)
	value = number
Increase
	Count(5).Increase is 6
	self = self + 1".SplitLines()), null!);
		new Type(this, new FileData(Base.Character, @"implement Number
from(number)
	value = number".SplitLines()), null!);
		new Type(this, new FileData(Base.Text, @"has Characters
Run
	value is not """"
+(other) returns Text
	return value
digits(number) returns Numbers
	if floor(number / 10) is 0
		return (number % 10)
	else
		return digits(floor(number / 10)) + number % 10".SplitLines()), null!);
		new Type(this, new FileData(Base.Log, @"has Text
Write(text)
	return Text
Write(number)
	return number".SplitLines()), null!);
		new Type(this, new FileData(Base.List, @"+(other) returns List
-(other) returns List
is(other) returns Boolean
*(other) returns List".SplitLines()), null!);
		new Type(this, new FileData(Base.Boolean, @"not returns Boolean
is(any) returns Boolean".SplitLines()), null!);
	}
}