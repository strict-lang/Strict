﻿using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class MutableTests : TestExpressions
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private MethodExpressionParser parser = null!;

	[Test]
	public void MutableMemberConstructorWithType()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableMemberConstructorWithType), "has something Mutable(Number)",
					"Add(input Count) Number",
					"\tlet result = something + input")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].IsMutable, Is.True);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void MutableMethodParameterWithType()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableMethodParameterWithType), "has something Character",
					"Add(input Mutable(Number)) Number",
					"\tlet result = input + something")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].Parameters[0].IsMutable,
			Is.True);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void MutableMemberWithTextType()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableMemberWithTextType), "has something Mutable(Text)",
					"Add(input Count) Text",
					"\tlet result = input + something")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());
	}

	[Test]
	public void MutableVariablesWithSameImplementationTypeShouldUseSameType()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(MutableVariablesWithSameImplementationTypeShouldUseSameType), "has unused Number",
				"UnusedMethod Number",
				"\tlet first = Mutable(5)",
				"\tlet second = Mutable(6)",
				"\tfirst + second")).ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0].ReturnType.Name, Is.EqualTo(Base.Mutable + "(TestPackage." + Base.Number + ")"));
		Assert.That(body.Expressions[0].ReturnType, Is.EqualTo(body.Expressions[1].ReturnType));
	}
}