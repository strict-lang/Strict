using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Moq;
using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace Strict.LanguageServer.Tests;

public sealed class TextDocumentSynchronizerTests
{
	private TextDocumentSynchronizer handler = null!;
	private static readonly DocumentUri URI = new("", "", "test.strict", "", "");

	[SetUp]
	public void Setup()
	{
		var windowMock = new Mock<IWindowLanguageServer>();
		windowMock.Setup(expression => expression.SendNotification(It.IsAny<string>()));
		var languageMock = new Mock<ILanguageServer>();
		languageMock.Setup(expression => expression.Window).Returns(windowMock.Object);
		handler = new TextDocumentSynchronizer(languageMock.Object);
		handler.DocumentManager.AddOrUpdate(URI, "let bla = 5");
	}

	[Test]
	public async Task HandleOpenTextDocumentAsync()
	{
		await handler.Handle(
			new DidOpenTextDocumentParams
			{
				TextDocument = new TextDocumentItem
				{
					Text = "let bla = 5", LanguageId = "strict", Uri = URI
				}
			}, new CancellationToken());
		Assert.That(handler.DocumentManager.Get(URI), Is.EqualTo(new[] { "let bla = 5" }));
	}

	private static IEnumerable<TestCaseData> TextDocumentChangeCases
	{
		//ncrunch: no coverage start
		get
		{
			yield return new TestCaseData(new Range(0, 0, 0, 0), "\t", new[] { "\tlet bla = 5" });
			yield return new TestCaseData(new Range(0, 10, 0, 10), "5", new[] { "let bla = 55" });
			yield return new TestCaseData(new Range(0, 0, 0, 11), "let bla = 15",
				new[] { "let bla = 15" });
			yield return new TestCaseData(new Range(0, 0, 0, 4), "", new[] { "bla = 5" });
			yield return new TestCaseData(new Range(1, 0, 1, 0), "let something = 5",
				new[] { "let bla = 5", "let something = 5" });
		}
		//ncrunch: no coverage end
	}

	[TestCaseSource(nameof(TextDocumentChangeCases))]
	public async Task HandleChangeTextDocumentAsync(Range range, string text, string[] expected)
	{
		await handler.Handle(
			new DidChangeTextDocumentParams
			{
				TextDocument = new OptionalVersionedTextDocumentIdentifier { Uri = URI },
				ContentChanges = new Container<TextDocumentContentChangeEvent>(
					new TextDocumentContentChangeEvent { Range = range, Text = text })
			}, new CancellationToken());
		Assert.That(handler.DocumentManager.Get(URI), Is.EqualTo(expected));
	}
}