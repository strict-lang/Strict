using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

//ncrunch: no coverage start
public static class BaseSelectors
{
	public static readonly TextDocumentSelector StrictDocumentSelector =
		new(new TextDocumentFilter { Pattern = "**/*" + Type.Extension });
}