using OmniSharp.Extensions.LanguageServer.Protocol.Models;

namespace Strict.LanguageServer;

//ncrunch: no coverage start
public static class BaseSelectors
{
	public static readonly TextDocumentSelector StrictDocumentSelector =
		new(new TextDocumentFilter { Pattern = "**/*" + Language.Type.Extension });
}