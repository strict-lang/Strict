using System.Threading.Tasks;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace Strict.Language.Tests;

public static class Program
{
	public static Task Main()
	{
		return new RepositoriesTests().LoadStrictBaseTypesJust10TimesWithDisabledCache();
		//var config = ManualConfig.Create(DefaultConfig.Instance);
		//config.Options = ConfigOptions.DisableOptimizationsValidator;
		//BenchmarkRunner.Run<RepositoriesTests>(config);
	}
}