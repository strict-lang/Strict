using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ValueInstanceTests
{
	[SetUp]
	public void CreateNumber() => numberType = TestPackage.Instance.GetType(Base.Number);

	private Type numberType = null!;

	[Test]
	public void ToStringShowsTypeAndValue() =>
		Assert.That(new ValueInstance(numberType, 42).ToString(), Is.EqualTo("Number:42"));

	[Test]
	public void CompareTwoNumbers() =>
		Assert.That(new ValueInstance(numberType, 42), Is.EqualTo(new ValueInstance(numberType, 42)));

	[Test]
	public void CompareNumberToText() =>
		Assert.That(new ValueInstance(numberType, 5),
			Is.Not.EqualTo(new ValueInstance(TestPackage.Instance.GetType(Base.Text), "5")));

	[Test]
	public void CompareLists()
	{
		var list = new ValueInstance(TestPackage.Instance.GetListImplementationType(numberType),
			new[] { 1, 2, 3 });
		Assert.That(list,
			Is.EqualTo(new ValueInstance(TestPackage.Instance.GetListImplementationType(numberType),
				new[] { 1, 2, 3 })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(TestPackage.Instance.GetListImplementationType(numberType),
				new[] { 1, 2, 1 })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(TestPackage.Instance.GetListImplementationType(numberType),
				new[] { 1, 2, 3, 4 })));
	}

	[Test]
	public void CompareDictionaries()
	{
		var list = new ValueInstance(
			TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
			new Dictionary<int, int> { { 1, 2 } });
		Assert.That(list,
			Is.EqualTo(new ValueInstance(
				TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
				new Dictionary<int, int> { { 1, 2 } })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(
				TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
				new Dictionary<int, int> { { 2, 2 } })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(
				TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
				new Dictionary<int, int> { { 1, 3 } })));
		Assert.That(list,
			Is.Not.EqualTo(new ValueInstance(
				TestPackage.Instance.GetDictionaryImplementationType(numberType, numberType),
				new Dictionary<int, int> { { 1, 3 }, { 2, 2 } })));
	}

	[Test]
	public void CompareTypeContainingNumber()
	{
		using var t =
			new Type(TestPackage.Instance,
					new TypeLines(nameof(CompareTypeContainingNumber),
						new[] { "has number", "Run Boolean", "\tnumber is 42" })).
				ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(new ValueInstance(t, 42), Is.EqualTo(new ValueInstance(numberType, 42)));
	}
}