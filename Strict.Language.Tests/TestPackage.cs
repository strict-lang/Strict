namespace Strict.Language.Tests;

/// <summary>
/// Helper context to provide a bunch of helper types to make tests work.
/// </summary>
public class TestPackage : Package
{
	public TestPackage() : base(nameof(TestPackage))
	{
		var anyType = new Type(this, new TypeLines(Base.Any, "is(other) Boolean"));
		new Type(this, new TypeLines(Base.Boolean, "not Boolean", "\tfalse", "is(any) Boolean", "\ttrue")).ParseMembersAndMethods(null!);
		anyType.ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Number,
				"+(other) Number",
				"\tvalue + other",
				"*(other) Number",
				"\tvalue * other",
				"-(other) Number",
				"\tvalue - other",
				"/(other) Number",
				"\tvalue / other",
				"Floor Number",
				"\tvalue - value % 1")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Count, "implement Number", "Increment",
				"\tCount(5).Increment is 6", "\tvalue = value + 1")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Character, "implement Number", "from(number)", "\tvalue = number")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Text,
				"has Characters",
				"Run",
				"\tvalue is not \"\"",
				"+(other) Text",
				"\treturn value",
				"digits(number) Numbers",
				"\tif floor(number / 10) is 0",
				"\t\treturn (number % 10)",
				"\telse",
				"\t\treturn digits(floor(number / 10)) + number % 10")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Log,
				"has Text",
				"Write(text)",
				"\tText",
				"Write(number)",
				"\tnumber")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.List,
				"+(other) List",
				"-(other) List",
				"is(other) Boolean",
				"*(other) List")).ParseMembersAndMethods(null!);
	}
}