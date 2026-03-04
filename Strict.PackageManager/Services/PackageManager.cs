using Grpc.Core;
using Strict.Language;

namespace Strict.PackageManager.Services;

//ncrunch: no coverage start
public sealed class PackageManager(ILogger<PackageManager> logger)
	: Strict.PackageManager.PackageManager.PackageManagerBase
{
	public override async Task<EmptyReply> DownloadPackage(PackageDownloadRequest request,
		ServerCallContext context)
	{
		var parts = request.PackageUrl.Split('/');
		var org = parts[3];
		var localCachePath = Path.Combine(CacheFolder, org, request.PackageName);
		logger.LogTrace("Service invoked " + request.PackageUrl + " " + request.PackageName + " " +
			request.TargetPath);
		if (!Directory.Exists(localCachePath))
			Directory.CreateDirectory(localCachePath);
		using var downloader = new GitHubStrictDownloader(org, request.PackageName);
		await downloader.DownloadFiles(localCachePath, context.CancellationToken);
		if (!string.IsNullOrWhiteSpace(request.TargetPath) &&
			!request.TargetPath.Equals(localCachePath, StringComparison.OrdinalIgnoreCase))
			CopyToTargetPath(localCachePath, request.TargetPath);
		return new EmptyReply();
	}

	private static string CacheFolder =>
		Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
			StrictPackages);
	private const string StrictPackages = nameof(StrictPackages);

	private static void CopyToTargetPath(string sourcePath, string targetPath)
	{
		if (!Directory.Exists(targetPath))
			Directory.CreateDirectory(targetPath);
		foreach (var file in Directory.GetFiles(sourcePath))
			File.Copy(file, Path.Combine(targetPath, Path.GetFileName(file)), true);
	}
}