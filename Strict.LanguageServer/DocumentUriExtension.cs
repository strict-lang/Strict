using OmniSharp.Extensions.LanguageServer.Protocol;
using Uri = System.Uri;

namespace Strict.LanguageServer;

public static class DocumentUriExtension
{
	public static string GetFolderName(this string path) => path.Split('/')[^2];
	public static string GetFileName(this string path) => path.Split("/")[^1].Split('.')[0];

public static string ToFileSystemPath(this string path)
	{
		if (path.Length >= 3 && path[0] == '/' && char.IsLetter(path[1]) && path[2] == ':')
			path = path[1..];
		return path.Replace('/', Path.DirectorySeparatorChar);
	}

	public static string ToLocalFile(this DocumentUri uri)
	{
		try
		{
			var file = uri.ToUri();
			if (file.IsFile)
				return NormalizeLocalPath(file.LocalPath);
		}
		catch
		{
		}
		return NormalizeLocalPath(Uri.UnescapeDataString(uri.Path));
	}

	private static string NormalizeLocalPath(string path)
	{
		path = Uri.UnescapeDataString(path);
		if (path.Length > 2 && path[0] == '/' && char.IsLetter(path[1]) && path[2] == ':')
			path = path[1..];
		return path.Replace('/', Path.DirectorySeparatorChar);
	}

	public static string GetFolderNameFromFile(this string localPath) =>
		Path.GetFileName(Path.GetDirectoryName(localPath) ?? localPath);
}