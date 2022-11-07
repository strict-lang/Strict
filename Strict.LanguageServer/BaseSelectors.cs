using OmniSharp.Extensions.LanguageServer.Protocol.Models;

namespace Strict.LanguageServer
{
	public static class BaseSelectors
	{
		public static readonly DocumentSelector StrictDocumentSelector = new(
			new DocumentFilter { Pattern = "**/*.strict" });
	}
}
