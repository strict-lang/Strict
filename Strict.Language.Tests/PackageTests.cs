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
	public void TearDown() => mainPackage.Dispose();

	[Test]
	public void NoneIsAlwaysKnown()
	{
		var emptyPackage = new Package(nameof(NoneIsAlwaysKnown));
		Assert.That(emptyPackage.FindType(Type.None), Is.Not.Null);
		Assert.That(emptyPackage.FindType(nameof(NoneIsAlwaysKnown)), Is.Null);
	}

	[Test]
	public void DependencyPackageTypeIsFoundBeforeOtherChildPackages()
	{
		new Type(new Package(mainPackage, "Unrelated"), new TypeLines("Shared", "Run"));
		var dependency = new Package(mainPackage, "Dependency");
		var expected = new Type(dependency, new TypeLines("Shared", "Run"));
		var user = new Package(mainPackage, "User");
		user.automaticallyLoadedDependencyPackages.Add(dependency);
		Assert.That(user.FindType("Shared"), Is.EqualTo(expected));
	}

	[Test]
	public void TypeAddedLaterWinsOverForeignTypeFoundBefore()
	{
		var foreign = new Type(new Package(mainPackage, "Foreign"), new TypeLines("Later", "Run"));
		Assert.That(new Type(subPackage, new TypeLines("Asker", "Run")).FindType("Later"),
			Is.EqualTo(foreign));
		var later = new Type(subPackage, new TypeLines("Later", "Run"));
		Assert.That(new Type(subPackage, new TypeLines("SecondAsker", "Run")).FindType("Later"),
			Is.EqualTo(later));
	}

	[Test]
	public void TypesCanBeEnumeratedWhileATypeIsAdded()
	{
		foreach (var _ in subPackage.Types)
			if (subPackage.FindDirectType("Added") == null)
				new Type(subPackage, new TypeLines("Added", "Run"));
		Assert.That(subPackage.FindDirectType("Added"), Is.Not.Null);
	}

	[Test]
	public void AnyMethodsComeFromTheAnyOfTheOwnPackageTree()
	{
		var parser = new MethodExpressionParser();
		using var testPackageType = new Type(TestPackage.Instance, new TypeLines("AnyUser", "Run")).
			ParseMembersAndMethods(parser);
		Assert.That(testPackageType.AvailableMethods, Does.Not.ContainKey("Hello"));
		var ownTree = new Package("OwnAnyTree");
		new Type(ownTree, new TypeLines(Type.Any, "from", "to Type", "to Text", "Hello Number")).
			ParseMembersAndMethods(parser);
		Assert.That(new Type(ownTree, new TypeLines("Probe", "Run")).ParseMembersAndMethods(parser).
			AvailableMethods, Does.ContainKey("Hello"));
		ownTree.Unload();
	}

	[Test]
	public void IsPrivateNameCheckShouldReturnNull() =>
		Assert.That(new Package(nameof(IsPrivateNameCheckShouldReturnNull)).FindType("isPrivate"),
			Is.Null);

	[Test]
	public void RootPackageToStringShouldNotCrash()
	{
		Assert.That(mainType.Package.Parent.FullName, Is.Empty);
		Assert.That(mainType.Package.Parent.FindType(Type.None)?.Name, Is.EqualTo(Type.None));
		Assert.That(mainPackage.Parent.GetPackage(), Is.Null);
	}

	[Test]
	public void GetFullNames()
	{
		Assert.That(mainPackage.FullName, Is.EqualTo(nameof(PackageTests)));
		Assert.That(mainType.FullName,
			Is.EqualTo(nameof(PackageTests) + Context.ParentSeparator + mainType.Name));
		Assert.That(subPackage.FullName,
			Is.EqualTo(nameof(PackageTests) + Context.ParentSeparator + nameof(subPackage)));
		Assert.That(privateSubType.FullName,
			Is.EqualTo(nameof(PackageTests) + Context.ParentSeparator + nameof(subPackage) +
				Context.ParentSeparator + privateSubType.Name));
		Assert.That(publicSubType.FullName,
			Is.EqualTo(nameof(PackageTests) + Context.ParentSeparator + nameof(subPackage) +
				Context.ParentSeparator + publicSubType.Name));
	}

	[Test]
	public void PrivateTypesCanOnlyBeFoundInPackageTheyAreIn()
	{
		Assert.That(mainType.GetType(publicSubType.Name), Is.EqualTo(publicSubType));
		Assert.Throws<Package.PrivateTypesAreOnlyAvailableInItsPackage>(() =>
			mainPackage.GetType(privateSubType.FullName));
		Assert.Throws<Package.PrivateTypesAreOnlyAvailableInItsPackage>(() =>
			mainPackage.GetType(nameof(TestPackage) + Context.ParentSeparator + nameof(PackageTests) +
				Context.ParentSeparator + privateSubType.Name));
	}

	[Test]
	public void FindSubTypeBothWays()
	{
		Assert.That(mainType.GetType(publicSubType.FullName), Is.EqualTo(publicSubType));
		Assert.That(publicSubType.GetType(mainType.FullName), Is.EqualTo(mainType));
	}

	[Test]
	public void FindPackage() =>
		Assert.That(mainPackage.Find(subPackage.FullName), Is.EqualTo(subPackage));

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

	[TestCase("/")]
	[TestCase("Strict/")]
	public void FullNameWithoutTypeNameIsNoType(string fullName) =>
		Assert.That(mainPackage.FindFullType(fullName), Is.Null);

	[Test]
	public void ContextNameMustNotContainSpecialCharactersOrNumbers()
	{
		Assert.That(() => new Type(mainPackage, new TypeLines("MyClass123")),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());
		Assert.That(() => new Package(mainPackage, "$%"),
			Throws.InstanceOf<Context.PackageNameMustBeAWordWithoutSpecialCharacters>());
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
		try
		{
			expressionParser.CreateType();
			using var strictPackage = await new Repositories(expressionParser).LoadStrictPackage();
			Assert.That(mainPackage.GetType(Type.Number),
				Is.EqualTo(strictPackage.GetType(Type.Number)).Or.EqualTo(subPackage.GetType(Type.Number)));
			Assert.That(mainPackage.GetType(Type.Character),
				Is.Not.EqualTo(mainPackage.FindType(Type.Any)));
		}
		finally
		{
			expressionParser.TearDown();
		}
	}

	[Test]
	public async Task ListOfSameNamedTypeInOtherPackageUsesThatType()
	{
		var parser = new MethodExpressionParser();
		var imageProcessing =
			await new Repositories(parser).LoadStrictPackage("Strict/ImageProcessing");
		var color = new Type(new Package((Package)imageProcessing.Parent, "OtherColors"),
			new TypeLines("Color", "has Red Number", "has Green Number", "Reds Colors",
				"\t(Color(1, 2))")).ParseMembersAndMethods(parser);
		Assert.That(color.Methods[0].ReturnType.GetFirstImplementation(), Is.SameAs(color));
		Assert.That(imageProcessing.GetType("Colors").FilePath,
			Is.EqualTo(color.GetType(Type.List).FilePath));
		color.Package.Unload();
	}

	[Test]
	public void SameNamedTypesOfTwoRootPackagesGetTheirOwnGenericImplementations()
	{
		var first = new Type(new Package("FirstWidgets"), new TypeLines("Widget", "Run"));
		var second = new Type(new Package("SecondWidgets"), new TypeLines("Widget", "Run"));
		var list = TestPackage.Instance.GetType(Type.List);
		var dictionary = TestPackage.Instance.GetType(Type.Dictionary);
		Assert.That(list.GetGenericImplementation(first).GetFirstImplementation(), Is.SameAs(first));
		Assert.That(list.GetGenericImplementation(second).GetFirstImplementation(), Is.SameAs(second));
		Assert.That(dictionary.GetGenericImplementation(first, first).ImplementationTypes,
			Is.EqualTo(new[] { first, first }));
		Assert.That(dictionary.GetGenericImplementation(first, second).ImplementationTypes,
			Is.EqualTo(new[] { first, second }));
		first.Package.Unload();
		second.Package.Unload();
	}

	/// <summary>
	/// Can be used to profile and optimize the GetType performance by doing it many times
	/// </summary>
	[Test]
	public void LoadingTypesOverAndOverWillAlwaysQuicklyReturnSame()
	{
		var otherMainPackage = new Package(nameof(LoadingTypesOverAndOverWillAlwaysQuicklyReturnSame));
		for (var index = 0; index < 1000; index++)
			if (otherMainPackage.FindType(mainType.Name)!.Name != mainType.Name)
				throw new AssertionException("FindType=" + //ncrunch: no coverage
					otherMainPackage.FindType(mainType.Name) + " didn't find " + mainType);
	}

	[Test]
	[Category("Slow")]
	public void LoadingStrictPackagesInParallelDoesNotFail()
	{
		var tasks = Enumerable.Range(0, 8).Select(async index =>
		{
			var typeSuffix = ((char)('A' + index)).ToString();
			var parser = new MethodExpressionParser();
			var repositories = new Repositories(parser);
			using var package = await repositories.LoadStrictPackage("Strict/ImageProcessing");
			using var testType = new Type(package, new TypeLines(
				nameof(LoadingStrictPackagesInParallelDoesNotFail) + typeSuffix,
				// @formatter: off
				"has number", "Run Number", "\tconstant width = 80", "\tconstant height = 45",
				"\tmutable image = Image(Size(width, height))", "\tfor image.Size",
				"\t\timage.Colors(index) = Color(0.25, 0.25, 0.25)", "\tmutable count = 0",
				"\tfor image.Size", "\t\tif image.Colors(index) is Color(0.25, 0.25, 0.25)",
				"\t\t\tcount = count + 1", "\tcount")).ParseMembersAndMethods(parser);
			// @formatter: on
			var runMethod = testType.Methods.Single(method => method.Name == Method.Run);
			return runMethod.GetBodyAndParseIfNeeded().ToString();
		});
		Assert.That(async () => await Task.WhenAll(tasks), Throws.Nothing);
	}
}