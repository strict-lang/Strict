namespace Strict.LanguageServer;

public static class DocumentUriExtension
{
	public static string GetFolderName(this string path) => path.Split('/')[^2];
	public static string GetFileName(this string path) => path.Split("/")[^1].Split('.')[0];
}