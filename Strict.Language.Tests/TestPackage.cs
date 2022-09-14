namespace Strict.Language.Tests;

/// <summary>
///   Helper context to provide a bunch of helper types to make tests work.
/// </summary>
public class TestPackage : Package
{
	public TestPackage() : base(nameof(TestPackage))
	{
		// @formatter:off
		var anyType = new Type(this, new TypeLines(Base.Any,
			"is(other) Boolean", "not(other) Boolean"));
		new Type(this, new TypeLines(Base.Boolean,
			"not Boolean",
			"\tvalue ? false else true",
			"is(other) Boolean",
			"\tvalue is other",
			"and(other) Boolean",
			"\tvalue and other ? true else false",
			"or(other) Boolean",
			"\tvalue or other ? false else true",
			"xor(other) Boolean",
			"\t(value and other) or (not value and not other) ? false else true")).ParseMembersAndMethods(null!);
		anyType.ParseMembersAndMethods(null!);
		var numberType = new Type(this,
			new TypeLines(Base.Number,
				"is not(other) Boolean",
				"\tvalue != other",
				"+(other) Number",
				"\tvalue + other",
				"*(other) Number",
				"\tvalue * other",
				"-(other) Number",
				"\tvalue - other",
				"/(other) Number",
				"\tvalue / other",
				"Floor Number",
				"\tvalue - value % 1",
				"%(other) Number",
				"\tvalue % other",
				">(other) Boolean",
				"\tvalue > other",
				"<(other) Boolean",
				"\tvalue < other",
				"^(other) Number",
				"\tvalue ^ other",
				"in(other) Number",
				"\tvalue",
				">=(other) Boolean",
				"\tvalue >= other",
				"<=(other) Boolean",
				"\tvalue <= other",
				"to Text",
				"\t\"\" + value"));
		new Type(this,
			new TypeLines(Base.Range,
				"implement Number",
				"has Start Number",
				"has End Number",
				"from(start Number, end Number)",
				"\tStart = start",
				"\tEnd = end",
				"Length",
				"\tRange(0, 5).Length is 5",
				"\tRange(2, 18).Length is 16",
				"\tEnd - Start")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines("HasLength",
				"Length Number")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Count,
				"implement Number",
				"implement HasLength",
				"Increment",
				"\tCount(5).Increment is 6",
				"\tvalue = value + 1",
				"Length Number",
				"\tnumber.Length")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Character,
				"implement Number",
				"from(number)",
				"\tvalue = number")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Text,
				"has Characters",
				"from(number)",
				"\tvalue = Character(number)",
				"Run",
				"\tvalue is not \"\"",
				"+(other) Text",
				"\treturn value",
				"digits(number) Numbers",
				"\tif floor(number / 10) is 0",
				"\t\treturn (number % 10)",
				"\telse",
				"\t\treturn digits(floor(number / 10)) + number % 10")).ParseMembersAndMethods(null!);
		numberType.ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Log,
				"has Text",
				"Write(text)",
				"\tText",
				"Write(number)",
				"\tnumber")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.List,
				"First",
				"+(other) List",
				"-(other) List",
				"is(other) Boolean",
				"*(other) List")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.File,
				"from(Text)",
				"Read Text",
				"Write(text)",
				"Delete",
				"Length Number")).ParseMembersAndMethods(null!);
		new Type(this,
			new TypeLines(Base.Type,
				"to Text")).ParseMembersAndMethods(null!);
	}
}