using System.Threading.Tasks;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace Strict.Language.Tests;

public static class Program
{
	/*
	public static void Main()
	{
		var config = ManualConfig.Create(DefaultConfig.Instance);
		config.Options = ConfigOptions.DisableOptimizationsValidator;
		BenchmarkRunner.Run<RepositoriesTests>(config);
	}
	*/

	public static async Task Main()
	{
		var tests = new RepositoriesTests();
		//await tests.LoadingZippedStrictBaseHundredTimes();
		//tests.LoadingAllStrictFilesWithoutAsyncHundredTimes();
		//tests.SortImplementsOneThousandTimesInParallel();
		//await tests.LoadStrictBaseTypesHundredTimes();
		await tests.LoadStrictBaseTypes();
	}
}