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
	/// Loads packages from url (like github) and caches it to disc for the current run. All locally
	/// cached packages and all types in them are always available for any .strict file in the Editor.
	/// If a type is not found, packages.strict.dev is asked if we can get a url (used here to load).
	/// </summary>
	/// <remarks>Everything in here is async, you can easily load many packages in parallel</remarks>
	public class Repositories
	{
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
				localPath = await DownloadAndExtractRepository(packageUrl, packageName); //ncrunch: no coverage
			return await LoadFromPath(localPath);
		}

		public class OnlyGithubDotComUrlsAreAllowedForNow : Exception { }

		//ncrunch: no coverage start, only called once per session and only if not on development machine
		private static async Task<string> DownloadAndExtractRepository(Uri packageUrl, string packageName)
		{
			if (!Directory.Exists(CacheFolder))
				Directory.CreateDirectory(CacheFolder);
			var targetPath = Path.Combine(CacheFolder, packageName);
			if (Directory.Exists(targetPath) && AlreadyDownloaded.Contains(targetPath))
				return targetPath;
			AlreadyDownloaded.Add(targetPath);
			using WebClient webClient = new WebClient();
			var localZip = Path.Combine(CacheFolder, packageName + ".zip");
			await webClient.DownloadFileTaskAsync(new Uri(packageUrl + "/archive/master.zip"), localZip);
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
			return targetPath;
		} 

		private static readonly List<string> AlreadyDownloaded = new List<string>();
		// ncrunch: no coverage end

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
		private static async Task<Package> CreatePackageFromFiles(string packagePath, string[] files,
			Package? parent = null)
		{
			// Main folder can be empty, other folders must contain at least one file to create a package
			if (parent != null && files.Length == 0)
				return parent; //ncrunch: no coverage, doesn't happen in nicely designed packages anyway
			var package = parent != null
				? new Package(parent, packagePath)
				: new Package(packagePath);
			var types = new List<Type>();
			foreach (var filePath in files)
				types.Add(new Type(package, Path.GetFileNameWithoutExtension(filePath), string.Empty));
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
		private static bool IsValidCodeDirectory(string directory) => Path.GetFileName(directory).IsWord();

		private static string DevelopmentFolder => Path.Combine(@"C:\code\GitHub\strict-lang");
		private static string CacheFolder =>
			Path.Combine( //ncrunch: no coverage, only downloaded and cached on non development machines
				Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
				StrictPackages);
		private const string StrictPackages = nameof(StrictPackages);
		public static readonly Uri StrictUrl = new Uri("https://github.com/strict-lang/Strict");
	}
}