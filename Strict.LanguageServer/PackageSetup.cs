using Strict.Language.Expressions;
using Strict.Language;

namespace Strict.LanguageServer;

public class PackageSetup
{
	private readonly Repositories repository;
	public PackageSetup() => repository = new Repositories(new MethodExpressionParser());
	public Task<Package> GetPackageAsync(string path) => repository.LoadFromPath(path);
}