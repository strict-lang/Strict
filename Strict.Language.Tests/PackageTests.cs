namespace Strict.Language.Tests;

public class PackageTests
{
	[SetUp]
	public void CreateContexts()
	{
		mainPackage = new Package(nameof(PackageTests));
		mainType = new Type(mainPackage, new TypeLines("Yolo", "Run"));
		subPackage = new Package(mainPackage, nameof(subPackage));
		privateSubType = new Type(subPackage, new TypeLines("secret", "Run"));
		publicSubType = new Type(subPackage, new TypeLines("FindMe", "Run"));
	}

	private Package mainPackage = null!;
	private Type mainType = null!;
	private Package subPackage = null!;
	private Type privateSubType = null!;
	private Type publicSubType = null!;

	[TearDown]
	public void TearDown() => ((Package)mainPackage.Parent).Remove(mainPackage);

	[Test]
	public void NoneIsAlwaysKnown()
	{
		var emptyPackage = new Package(nameof(NoneIsAlwaysKnown));
		Assert.That(emptyPackage.FindType(Base.None, emptyPackage), Is.Not.Null);
		Assert.That(emptyPackage.FindType(nameof(NoneIsAlwaysKnown), emptyPackage), Is.Null);
	}

	[Test]
	public void IsPrivateNameCheckShouldReturnNull() =>
		Assert.That(new Package(nameof(IsPrivateNameCheckShouldReturnNull)).FindType("isPrivate"),
			Is.Null);

	[Test]
	public void RootPackageToStringShouldNotCrash()
	{
		Assert.That(mainType.Package.Parent.ToString(), Is.Empty);
		Assert.That(mainType.Package.Parent.FindType(Base.None)?.Name, Is.EqualTo(Base.None));
		Assert.That(mainPackage.Parent.GetPackage(), Is.Null);
	}

	[Test]
	public void GetFullNames()
	{
		Assert.That(mainPackage.ToString(), Is.EqualTo(nameof(PackageTests)));
		Assert.That(mainType.ToString(), Is.EqualTo(nameof(PackageTests) + "." + mainType.Name));
		Assert.That(subPackage.ToString(),
			Is.EqualTo(nameof(PackageTests) + "." + nameof(subPackage)));
		Assert.That(privateSubType.ToString(),
			Is.EqualTo(nameof(PackageTests) + "." + nameof(subPackage) + "." + privateSubType.Name));
		Assert.That(publicSubType.ToString(),
			Is.EqualTo(nameof(PackageTests) + "." + nameof(subPackage) + "." + publicSubType.Name));
	}

	[Test]
	public void PrivateTypesCanOnlyBeFoundInPackageTheyAreIn()
	{
		Assert.That(mainType.GetType(publicSubType.Name), Is.EqualTo(publicSubType));
		Assert.Throws<Package.PrivateTypesAreOnlyAvailableInItsPackage>(() =>
			mainPackage.GetType(privateSubType.ToString()));
		Assert.Throws<Package.PrivateTypesAreOnlyAvailableInItsPackage>(() =>
			mainPackage.GetType(nameof(TestPackage) + "." + nameof(PackageTests) + "." +
				privateSubType.Name));
	}

	[Test]
	public void FindSubTypeBothWays()
	{
		Assert.That(mainType.GetType(publicSubType.ToString()), Is.EqualTo(publicSubType));
		Assert.That(publicSubType.GetType(mainType.ToString()), Is.EqualTo(mainType));
	}

	[Test]
	public void FindPackage() =>
		Assert.That(mainPackage.Find(subPackage.ToString()), Is.EqualTo(subPackage));

	[Test]
	public void FindUnknownPackage() =>
		Assert.That(mainPackage.Find(nameof(FindUnknownPackage)), Is.Null);

	[Test]
	public void RemovePackage()
	{
		mainPackage.Remove(mainType);
		Assert.That(mainPackage.FindDirectType(publicSubType.Name), Is.Null);
	}

	[Test]
	public void FindingFullTypeRequiresFullName() =>
		Assert.Throws<Package.FullNameMustContainPackageAndTypeNames>(() =>
			mainPackage.FindFullType(publicSubType.Name));

	[Test]
	public void ContextNameMustNotContainSpecialCharactersOrNumbers()
	{
		Assert.That(() => new Type(mainPackage, new TypeLines("MyClass123")),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());
		Assert.That(() => new Package(mainPackage, "$%"),
			Throws.InstanceOf<
				Context.PackageNameMustBeAWordWithoutSpecialCharacters>());
	}

	[TestCase("Hello-World")]
	[TestCase("MyPackage2022")]
	[TestCase("Math-Algebra-2")]
	public void PackageNameCanContainNumbersOrHyphenInMiddleOrEnd(string name) =>
		Assert.That(() => new Package(mainPackage, name),
			Does.Not.InstanceOf<Context.PackageNameMustBeAWordWithoutSpecialCharacters>());

	[TestCase("1Pack")]
	[TestCase("-Pack")]
	[TestCase("Pack,")]
	[TestCase("Pack(*^&*)")]
	public void PackageNameMustNotContainNumbersOrHyphenInBeginning(string name) =>
		Assert.That(() => new Package(mainPackage, name),
			Throws.InstanceOf<Context.PackageNameMustBeAWordWithoutSpecialCharacters>());

	[Test]
	public async Task LoadTypesFromOtherPackage()
	{
		var expressionParser = new ExpressionParserTests();
		expressionParser.CreateType();
		using var strictPackage = await new Repositories(expressionParser).LoadStrictPackage();
		Assert.That(mainPackage.GetType(Base.App),
			Is.EqualTo(strictPackage.GetType(Base.App)).Or.EqualTo(subPackage.GetType(Base.App)));
		Assert.That(mainPackage.GetType(Base.Character),
			Is.Not.EqualTo(mainPackage.GetType(Base.App)));
		expressionParser.TearDown();
	}

	/// <summary>
	/// Can be used to profile and optimize the GetType performance by doing it many times
	/// </summary>
	[Test]
	public void LoadingTypesOverAndOverWillAlwaysQuicklyReturnSame()
	{
		var otherMainPackage =
			new Package(nameof(LoadingTypesOverAndOverWillAlwaysQuicklyReturnSame));
		for (var index = 0; index < 1000; index++)
			if (otherMainPackage.FindType(mainType.Name, otherMainPackage)!.Name != mainType.Name)
				throw new AssertionException("FindType=" + //ncrunch: no coverage
					otherMainPackage.FindType(mainType.Name, otherMainPackage) + " didn't find " +
					mainType);
	}
}