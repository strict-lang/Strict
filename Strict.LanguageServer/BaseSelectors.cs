using OmniSharp.Extensions.LanguageServer.Protocol.Models;

namespace Strict.LanguageServer;

//ncrunch: no coverage start
public static class BaseSelectors
{
	public static readonly DocumentSelector StrictDocumentSelector =
		new(
			new DocumentFilter { Pattern = "**/*.strict" });
} //ncrunch: no coverage end