using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Threading.Tasks;

namespace Strict.Language
{
	/// <inheritdoc />
	/// <summary>
	/// In C# or Java called namespace or package as well, in Strict this is any code folder.
	/// </summary>
	public class Package : Context
	{
		public Package(string packageName) : base(RootForPackages, packageName) { }

		private static readonly Root RootForPackages = new Root();

		/// <summary>
		/// Contains all high level <see cref="Package"/>. Just contains the fallback None type (think
		/// void) and Boolean, has no parent and just contains all root children packages.
		/// </summary>
		private class Root : Package
		{
			public Root() : base(null!, string.Empty)
			{
				new Type(this, Base.None, "");
				new Type(this, Base.Boolean, "");
			}

			public override Type? FindType(string name, Package? searchingFromPackage = null,
				Type? searchingFromType = null) =>
				name == Base.None || name == Base.Boolean
					? base.FindType(name, searchingFromPackage, searchingFromType)
					: null;
		}

		public Package(Package parentPackage, string folderName) : base(parentPackage, folderName) { }

		internal void Add(Type type) => types.Add(type);
		private readonly List<Type> types = new List<Type>();

		public override Type? FindType(string name, Package? searchingFromPackage = null,
			Type? searchingFromType = null) =>
			types.Find(t => t.Name == name) ?? (name.Contains(".")
				? types.Find(t => t.ToString() == name)
				: null) ?? AbortIfTypeIsPrivate(name) ??
			Parent.FindType(name, this, searchingFromType) ??
			FindTypeInChildren(name, searchingFromPackage, searchingFromType);

		private static Type? AbortIfTypeIsPrivate(string name) =>
			char.IsLower(name.Split('.').Last()[0])
				? throw new PrivateTypesAreOnlyAvailableInItsPackage()
				: (Type?)null;

		public class PrivateTypesAreOnlyAvailableInItsPackage : Exception {}

		private Type? FindTypeInChildren(string name, Package? searchingFromPackage, Type? searchingFromType)
		{
			foreach (var child in Children)
				if (child != searchingFromType && child != searchingFromPackage)
				{
					var childType = child is Package
						? child.FindType(name, searchingFromPackage, searchingFromType)
						: child.Name == name || child.ToString() == name
							? child
							: null;
					if (childType != null)
						return (Type)childType;
				}
			return null;
		}

		public Type? FindDirectType(string name) => types.Find(t => t.Name == name);
		public Package GetSubPackage(string name) => (Package)Children.First(p => p.Name == name);

		/// <summary>
		/// Loads from url (like github) and caches it to disc (will only check for updates once per day)
		/// All locally cached packages and all their types in them are always available for any .strict
		/// file in the Editor. If a type is not found, packages.strict.dev is asked if we can get a url.
		/// </summary>
		public static async Task<Package> FromUrl(string packageUrl)
		{
			var uri = new Uri(packageUrl);
			if (uri.Host != "github.com" || string.IsNullOrEmpty(uri.AbsolutePath) ||
				// Allow other repositories as well, but put them in an empty main package name first
				!uri.AbsolutePath.StartsWith("/strict-lang/"))
				throw new OnlyGithubDotComUrlsAreAllowedForNow();
			var packageName = uri.AbsolutePath.Split('/').Last();
			var localPath = Path.Combine(DevelopmentFolder, packageName);
			if (!Directory.Exists(localPath))
				localPath = Path.Combine(CacheFolder, packageName); //ncrunch: no coverage
			if (!Directory.Exists(localPath))
				await DownloadAndExtractRepository(packageUrl, localPath, packageName); //ncrunch: no coverage
			return await FromDiskPath(localPath);
		}

		public class OnlyGithubDotComUrlsAreAllowedForNow : Exception { }

		//ncrunch: no coverage start, should only be called rarely if we are missing a cached package
		private static async Task DownloadAndExtractRepository(string packageUrl, string localPath, string packageName)
		{
			if (!Directory.Exists(CacheFolder))
				Directory.CreateDirectory(CacheFolder);
			using WebClient webClient = new WebClient();
			var localZip = Path.Combine(CacheFolder, packageName + ".zip");
			await webClient.DownloadFileTaskAsync(new Uri(packageUrl + "/archive/master.zip"), localZip);
			await Task.Run(() =>
			{
				ZipFile.ExtractToDirectory(localZip, CacheFolder);
				var masterDirectory = Path.Combine(CacheFolder, packageName + "-master");
				if (Directory.Exists(masterDirectory))
					Directory.Move(masterDirectory, localPath);
			});
		} // ncrunch: no coverage end

		private static Task<Package> FromDiskPath(string packagePath) =>
			CreatePackageFromFiles(packagePath, RootForPackages,
				Directory.GetFiles(packagePath, "*" + Type.Extension));

		/// <summary>
		/// Initially we need to create just empty types and then after they all have been created
		/// we will fill and load them, otherwise we could not use types within the package context.
		/// </summary>
		private static async Task<Package> CreatePackageFromFiles(string packagePath, Package parent,
			string[] files)
		{
			if (parent != RootForPackages && files.Length == 0)
				return null!;
			var package = new Package(parent, Path.GetFileName(packagePath));
			foreach (var filePath in files)
				new Type(package, Path.GetFileNameWithoutExtension(filePath), string.Empty);
			var tasks = new List<Task>();
			for (var index = 0; index < package.Children.Count; index++)
				tasks.Add(package.types[index].ParseFile(files[index]));
			foreach (var directory in Directory.GetDirectories(packagePath))
				tasks.Add(CreatePackageFromFiles(directory, package,
					Directory.GetFiles(directory, "*" + Type.Extension)));
			await Task.WhenAll(tasks);
			return package;
		}

		private static string DevelopmentFolder => Path.Combine(@"C:\code\GitHub\strict-lang");
		private static string CacheFolder =>
			Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
				StrictPackages);
		private const string StrictPackages = nameof(StrictPackages);
		public string LocalCachePath =>
			Path.Combine(CacheFolder, ToString().Replace('.', Path.DirectorySeparatorChar));
		public const string StrictUrl = "https://github.com/strict-lang/Strict";
	}
}