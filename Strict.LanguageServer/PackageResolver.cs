using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

public static class PackageResolver
{
	public static Package Resolve(Package root, string filePath)
	{
		var directory = Path.GetDirectoryName(filePath);
		if (string.IsNullOrEmpty(directory))
			return FromFolderName(root, Path.GetFileName(Path.GetDirectoryName(filePath) ?? root.Name));
		if (!string.IsNullOrEmpty(root.FolderPath) &&
			string.Equals(Path.GetFullPath(directory), Path.GetFullPath(root.FolderPath),
				StringComparison.OrdinalIgnoreCase))
			return root;
		var folders = new Stack<string>();
		var current = directory;
		while (!string.IsNullOrEmpty(current))
		{
			if (!string.IsNullOrEmpty(root.FolderPath) &&
				string.Equals(Path.GetFullPath(current), Path.GetFullPath(root.FolderPath),
					StringComparison.OrdinalIgnoreCase) || File.Exists(Path.Combine(current, "Boolean.strict")))
			{
				var resolved = root;
				var pathSoFar = current;
				foreach (var folder in folders)
				{
					pathSoFar = Path.Combine(pathSoFar, folder);
					resolved = resolved.Find(folder) ?? LoadOrCreate(resolved, folder, pathSoFar);
				}
				return resolved;
			}
			folders.Push(Path.GetFileName(current));
			current = Path.GetDirectoryName(current);
		}
		return FromFolderName(root, Path.GetFileName(directory));
	}

	private static Package FromFolderName(Package root, string folderName) =>
		string.Equals(folderName, root.Name, StringComparison.OrdinalIgnoreCase)
			? root
			: root.Find(folderName) ?? new Package(root, folderName);

	private static Package LoadOrCreate(Package parent, string folderName, string folderPath)
	{
		var existing = parent.Find(folderName);
		if (existing != null)
			return existing;
		if (!Directory.Exists(folderPath))
			return new Package(parent, folderName);
		var created = new Package(parent, folderPath);
		foreach (var file in Directory.GetFiles(folderPath, "*" + Type.Extension))
		{
			var typeName = Path.GetFileNameWithoutExtension(file);
			if (created.FindDirectType(typeName) != null)
				continue;
			try
			{
				new Type(created, new TypeLines(typeName, TypeLines.FromFile(file))).
					ParseMembersAndMethods(new MethodExpressionParser());
			}
			catch (Exception)
			{
				// Sibling files can fail; the requested file is synchronized next.
			}
		}
		return created;
	}
}
