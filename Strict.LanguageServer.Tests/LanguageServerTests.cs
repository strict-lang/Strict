using Moq;
using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using Strict.Language.Tests;

namespace Strict.LanguageServer.Tests;

public class LanguageServerTests
{
	protected static readonly DocumentUri URI = new("", "", "Test/Test" + Language.Type.Extension,
		"", "");
	protected TextDocumentSynchronizer textDocumentHandler = null!;
	protected Mock<ILanguageServer> languageServer = null!;

	[SetUp]
	public void MockHandlers()
	{
		var window = new Mock<IWindowLanguageServer>();
		window.Setup(expression => expression.SendNotification(It.IsAny<string>()));
		languageServer = new Mock<ILanguageServer>();
		languageServer.Setup(expression => expression.Window).Returns(window.Object);
		languageServer.Setup(expression => expression.TextDocument).
			Returns(new Mock<ITextDocumentLanguageServer>().Object);
		textDocumentHandler =
			new TextDocumentSynchronizer(languageServer.Object, new StrictDocument(), TestPackage.Instance);
		textDocumentHandler.Document.AddOrUpdate(URI, "constant bla = 5");
	}
}