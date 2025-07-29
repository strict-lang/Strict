namespace Strict.Language.Tests;

public static class Program
{
	//ncrunch: no coverage start
	public static async Task Main()
	{
		var tests = new RepositoriesTests();
		await tests.LoadingZippedStrictBaseHundredTimes();
		tests.LoadingAllStrictFilesWithoutAsyncHundredTimes();
		tests.SortImplementsOneThousandTimesInParallel();
		await tests.LoadStrictBaseTypesHundredTimes();
		await tests.LoadStrictBaseTypes();
	}
}