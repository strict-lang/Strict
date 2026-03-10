namespace Strict.Language.Tests;

public static class Program
{
	//ncrunch: no coverage start
	public static async Task Main()
	{
		var tests = new RepositoriesTests();
		tests.CreateRepositories();
		tests.LoadingAllStrictFilesWithoutAsyncHundredTimes();
		tests.SortImplementsOneThousandTimesInParallel();
		await tests.LoadStrictBaseTypesHundredTimes();
		tests.DisposeParserType();
		await tests.LoadStrictBaseTypes();
	}
}