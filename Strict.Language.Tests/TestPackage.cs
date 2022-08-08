namespace Strict.Language.Tests;

/// <summary>
/// Helper context to provide a bunch of helper types to make tests work.
/// </summary>
public class TestPackage : Package
{
	public TestPackage() : base(nameof(TestPackage))
	{
		var anyType = new Type(this, new TypeLines(Base.Any, "is(other) returns Boolean"));
		new Type(this, new TypeLines(Base.Boolean, "not returns Boolean", "\treturn false", "is(any) returns Boolean", "\treturn true")).ParseMembersAndMethods(null!);
		anyType.ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Number, "+(other) returns Number", "\treturn self + other",
				"*(other) returns Number", "\treturn self * other", "-(other) returns Number",
				"\treturn self - other", "/(other) returns Number", "\treturn self / other", "Floor returns Number", "\treturn value - value % 1")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Count, "implement Number", "from(number)", "\tvalue = number", "Increase",
				"\tCount(5).Increase is 6", "\tself = self + 1")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Character, "implement Number", "from(number)", "\tvalue = number")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Text, "has Characters", "Run", "\tvalue is not \"\"",
				"+(other) returns Text", "\treturn value", "digits(number) returns Numbers",
				"\tif floor(number / 10) is 0", "\t\treturn (number % 10)", "\telse",
				"\t\treturn digits(floor(number / 10)) + number % 10")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Log, "has Text", "Write(text)", "\treturn Text", "Write(number)",
				"\treturn number")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.List, "+(other) returns List", "-(other) returns List",
				"is(other) returns Boolean", "*(other) returns List")).ParseMembersAndMethods(null!);
	}
}