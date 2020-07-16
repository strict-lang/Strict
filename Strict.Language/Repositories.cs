using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Threading.Tasks;

namespace Strict.Language
{
	/// <summary>
	/// Loads packages from url (like github) and caches it to disc for the current and subsequent
	/// runs. Next time Repositories is created, we will check for outdated cache and delete the zip
	/// files to allow redownloading fresh files. All locally cached packages and all types in them
	/// are always available for any .strict file in the Editor. If a type is not found,
	/// packages.strict.dev is asked if we can get a url (used here to load).
	/// </summary>
	/// <remarks>Everything in here is async, you can easily load many packages in parallel</remarks>
	public class Repositories
	{
		/// <summary>
		/// Gets rid of any cached zip files (keeps the actual files for use) older than 1h, which will
		/// allow redownloading from github to get any changes, while still staying fast in local runs
		/// when there are usually no changes happening.
		/// </summary>
		public Repositories(ExpressionParser parser)
		{
			this.parser = parser;
			if (Directory.Exists(CacheFolder))
				//ncrunch: no coverage start, rarely happens
				foreach (var file in Directory.GetFiles(CacheFolder, "*.zip"))
					if (File.GetLastWriteTimeUtc(file) < DateTime.UtcNow.AddHours(-1))
						File.Delete(file);
		} //ncrunch: no coverage end

		private readonly ExpressionParser parser;

		public async Task<Package> LoadFromUrl(Uri packageUrl)
		{
			if (packageUrl.Host != "github.com" || string.IsNullOrEmpty(packageUrl.AbsolutePath) ||
				// Allow other repositories as well, but put them in an empty main package name first
				!packageUrl.AbsolutePath.StartsWith("/strict-lang/"))
				throw new OnlyGithubDotComUrlsAreAllowedForNow();
			var packageName = packageUrl.AbsolutePath.Split('/').Last();
			var localPath = Path.Combine(DevelopmentFolder, packageName);
			// For some reason on the CI server an empty folder is still created here with a .dotsettings file
			if (!Directory.Exists(localPath) || Directory.GetFiles(localPath).Length < 2)
				//ncrunch: no coverage start
				localPath = await DownloadAndExtractRepository(packageUrl, packageName);
			//ncrunch: no coverage end
			return await LoadFromPath(localPath);
		}

		public class OnlyGithubDotComUrlsAreAllowedForNow : Exception { }

		//ncrunch: no coverage start, only called once per session and only if not on development machine
		private static async Task<string> DownloadAndExtractRepository(Uri packageUrl,
			string packageName)
		{
			if (!Directory.Exists(CacheFolder))
				Directory.CreateDirectory(CacheFolder);
			var targetPath = Path.Combine(CacheFolder, packageName);
			if (Directory.Exists(targetPath) &&
				File.Exists(Path.Combine(CacheFolder, packageName + ".zip")))
				return targetPath;
			await DownloadAndExtract(packageUrl, packageName, targetPath);
			return targetPath;
		}

		private static async Task DownloadAndExtract(Uri packageUrl, string packageName,
			string targetPath)
		{
			var localZip = Path.Combine(CacheFolder, packageName + ".zip");
			File.CreateText(localZip).Close();
			using WebClient webClient = new WebClient();
			await webClient.DownloadFileTaskAsync(new Uri(packageUrl + "/archive/master.zip"),
				localZip);
			await Task.Run(() =>
			{
				ZipFile.ExtractToDirectory(localZip, CacheFolder, true);
				var masterDirectory = Path.Combine(CacheFolder, packageName + "-master");
				if (!Directory.Exists(masterDirectory))
					return;
				if (Directory.Exists(masterDirectory))
					new DirectoryInfo(targetPath).Delete(true);
				Directory.Move(masterDirectory, targetPath);
			});
		} //ncrunch: no coverage end

		public async Task<Package> LoadFromPath(string packagePath)
		{
			if (AlreadyLoadedPackages.TryGetValue(packagePath, out var loadedPackage))
				return loadedPackage;
			var newPackage = await CreatePackageFromFiles(packagePath,
				Directory.GetFiles(packagePath, "*" + Type.Extension));
			AlreadyLoadedPackages.Add(packagePath, newPackage);
			return newPackage;
		}

		private static readonly Dictionary<string, Package> AlreadyLoadedPackages =
			new Dictionary<string, Package>();

		/// <summary>
		/// Initially we need to create just empty types and then after they all have been created
		/// we will fill and load them, otherwise we could not use types within the package context.
		/// </summary>
		private async Task<Package> CreatePackageFromFiles(string packagePath, string[] files,
			Package? parent = null)
		{
			// Main folder can be empty, other folders must contain at least one file to create a package
			if (parent != null && files.Length == 0)
				//ncrunch: no coverage start, doesn't happen in nicely designed packages anyway
				return parent;
			//ncrunch: no coverage end
			return await CreatePackage(packagePath, files, parent);
		}

		private async Task<Package> CreatePackage(string packagePath, string[] files,
			Package? parent)
		{
			var package = parent != null
				? new Package(parent, packagePath)
				: new Package(packagePath);
			var types = new List<Type>();
			foreach (var filePath in files)
				types.Add(new Type(package, Path.GetFileNameWithoutExtension(filePath), parser));
			var tasks = new List<Task>();
			for (var index = 0; index < types.Count; index++)
				tasks.Add(types[index].ParseFile(files[index]));
			foreach (var directory in Directory.GetDirectories(packagePath))
				if (IsValidCodeDirectory(directory))
					tasks.Add(CreatePackageFromFiles(directory,
						Directory.GetFiles(directory, "*" + Type.Extension), package));
			await Task.WhenAll(tasks);
			return package;
		}

		/// <summary>
		/// In Strict only words are valid directory names = package names, no symbols (like .git, .hg,
		/// .vs or _NCrunch) or numbers or dot separators (like Strict.Compiler) are allowed.
		/// </summary>
		private static bool IsValidCodeDirectory(string directory) =>
			Path.GetFileName(directory).IsWord();

		private static string DevelopmentFolder => Path.Combine(@"C:\code\GitHub\strict-lang");
		private static string CacheFolder =>
			Path.Combine( //ncrunch: no coverage, only downloaded and cached on non development machines
				Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
				StrictPackages);
		private const string StrictPackages = nameof(StrictPackages);
		public static readonly Uri StrictUrl = new Uri("https://github.com/strict-lang/Strict");
	}
}