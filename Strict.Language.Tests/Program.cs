using System.Threading.Tasks;

namespace Strict.Language.Tests;

public static class Program
{
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