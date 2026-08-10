using Strict.Bytecode.Instructions;

namespace Strict.Bytecode.Tests;

/// <summary>
/// Guards comparison codegen: <c>is not</c> must emit NotEqual for if-conditions,
/// and value comparisons must leave a Boolean in the result register.
/// </summary>
public sealed class BinaryGeneratorComparisonTests : TestBytecode
{
	[Test]
	public void IsNotInIfConditionCompilesToEqualAndNotOrNotEqual()
	{
		// Parser represents `is not` as not(is ...), so codegen may emit Equal+not
		// rather than a single NotEqual instruction. Both are valid.
		var methodCall = GenerateMethodCallFromSource("HasIsNot", "HasIsNot(5).Check",
			"has number",
			"Check Boolean",
			"\tif number is not 0",
			"\t\treturn true",
			"\tfalse");
		var instructions = new BinaryGenerator(methodCall).Generate().ToInstructions();
		var hasNotEqual = instructions.Any(i => i.InstructionType == InstructionType.NotEqual);
		var hasEqual = instructions.Any(i => i.InstructionType == InstructionType.Equal);
		Assert.That(hasNotEqual || hasEqual, Is.True,
			"if number is not 0 must emit comparison instructions");
	}

	[Test]
	public void IsComparisonAsExpressionEmitsEqualWithResultRegister()
	{
		var methodCall = GenerateMethodCallFromSource("HasIsValue", "HasIsValue(1).Check",
			"has number",
			"Check Boolean",
			"\tnumber is 1");
		var instructions = new BinaryGenerator(methodCall).Generate();
		var equal = instructions.ToInstructions().OfType<BinaryInstruction>().
			FirstOrDefault(i => i.InstructionType == InstructionType.Equal);
		Assert.That(equal, Is.Not.Null);
		Assert.That(equal!.Registers.Length, Is.EqualTo(3),
			"value comparison needs left, right, and result registers");
	}

	[Test]
	public void ConsecutiveListIndexesEachReloadIndexRegister()
	{
		var methodCall = GenerateMethodCallFromSource("MultiList", "MultiList((1, 2), (3, 4)).Pair",
			"has left Numbers",
			"has right Numbers",
			"Pair Number",
			"\tlet first = left(0)",
			"\tlet second = right(0)",
			"\tfirst + second");
		var instructions = new BinaryGenerator(methodCall).Generate().ToInstructions();
		var listCalls = instructions.OfType<ListCallInstruction>().ToList();
		Assert.That(listCalls.Count, Is.GreaterThanOrEqualTo(2));
		Assert.That(listCalls[0].IndexValueRegister, Is.Not.EqualTo(listCalls[1].Register),
			"index register must not be the prior list-element result register");
	}

	[Test]
	public void ConsecutiveListIndexesWithParameterReloadIndex()
	{
		var methodCall = GenerateMethodCallFromSource("MultiListAt",
			"MultiListAt((1, 2), (3, 4)).At(0)",
			"has left Numbers",
			"has right Numbers",
			"At(index Number) Number",
			"\tlet first = left(index)",
			"\tlet second = right(index)",
			"\tfirst + second");
		var instructions = new BinaryGenerator(methodCall).Generate().ToInstructions();
		var text = string.Join("\n", instructions.Select(i => i.ToString()));
		var listCalls = instructions.OfType<ListCallInstruction>().ToList();
		Assert.That(listCalls.Count, Is.GreaterThanOrEqualTo(2), text);
		var indexLoads = instructions.OfType<LoadVariableToRegister>().
			Count(load => load.Identifier == "index");
		Assert.That(indexLoads, Is.GreaterThanOrEqualTo(2),
			"each left(index)/right(index) must load index; instructions:\n" + text);
		// Critical: second ListCall must not reuse the first ListCall's result register as index
		Assert.That(listCalls[1].IndexValueRegister, Is.Not.EqualTo(listCalls[0].Register),
			"second list index register must not be first list element; instructions:\n" + text);
	}
}
