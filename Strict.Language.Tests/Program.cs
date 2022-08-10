using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace Strict.Language.Tests;

public static class Program
{
	public static void Main()
	{
		var config = ManualConfig.Create(DefaultConfig.Instance);
		config.Options = ConfigOptions.DisableOptimizationsValidator;
		BenchmarkRunner.Run<RepositoriesTests>(config);
	}

	//public static Task Main() => new RepositoriesTests().LoadStrictBaseTypesHundredTimes();
}