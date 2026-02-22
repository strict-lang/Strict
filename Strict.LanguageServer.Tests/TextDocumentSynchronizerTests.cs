using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Strict.Language.Tests;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace Strict.LanguageServer.Tests;

public sealed class TextDocumentSynchronizerTests : LanguageServerTests
{
	[SetUp]
	public void MultiLineSetup()
	{
		textDocumentHandler.Document.AddOrUpdate(MultiLineURI, "has number", "Add(num Number) Number",
			"\tnum + number");
		textDocumentHandler.Document.InitializeContent(MultiLineURI);
	}

	private static readonly DocumentUri MultiLineURI = new("", "", "Test/MultiLine.strict", "", "");
	private static IEnumerable<TestCaseData> TextDocumentChangeCases
	{
		//ncrunch: no coverage start
		get
		{
			yield return new TestCaseData(new Range(0, 0, 0, 0), "\t", new[] { "\tconstant bla = 5" });
			yield return new TestCaseData(new Range(0, 15, 0, 15), "5", new[] { "constant bla = 55" });
			yield return new TestCaseData(new Range(0, 0, 0, 16), "constant bla = 15",
				new[] { "constant bla = 15" });
			yield return new TestCaseData(new Range(0, 0, 0, 9), "", new[] { "bla = 5" });
			yield return new TestCaseData(new Range(1, 0, 1, 0), "constant something = 5",
				new[] { "constant bla = 5", "constant something = 5" });
		} //ncrunch: no coverage end
	}
	private static IEnumerable<TestCaseData> MultiLineTextDocumentChanges
	{
		//ncrunch: no coverage start
		get
		{
			yield return new TestCaseData(new Range(1, 22, 2, 13), "",
				new[] { "has number", "Add(num Number) Number" });
			yield return new TestCaseData(new Range(1, 12, 2, 13), "",
				new[] { "has number", "Add(num Numb" });
			yield return new TestCaseData(new Range(0, 0, 2, 13), "",
				new[] { "" });
			yield return new TestCaseData(new Range(0, 1, 2, 13), "",
				new[] { "h" });
			yield return new TestCaseData(new Range(1, 1, 2, 13), "", new[] { "has number", "A" });
			yield return new TestCaseData(new Range(1, 3, 2, 14), "", new[] { "has number", "Add" });
			yield return new TestCaseData(new Range(0, 3, 1, 3), "", new[] { "has(num Number) Number", "\tnum + number" });
			yield return new TestCaseData(new Range(0, 3, 2, 3), "", new[] { "hasm + number" });
			yield return new TestCaseData(new Range(3, 0, 4, 2), "NextMethod\n\t5",
				new[] { "has number", "Add(num Number) Number", "\tnum + number", "NextMethod", "\t5" });
			yield return new TestCaseData(new Range(2, 10, 2, 10), Environment.NewLine,
				new[] { "has number", "Add(num Number) Number", "\tnum + number", "" });
		} //ncrunch: no coverage end
	}

	[TestCaseSource(nameof(TextDocumentChangeCases))]
	public async Task HandleChangeTextDocumentAsync(Range range, string text, string[] expected)
	{
		await textDocumentHandler.Handle(
			new DidChangeTextDocumentParams
			{
				TextDocument = new OptionalVersionedTextDocumentIdentifier { Uri = URI },
				ContentChanges = new Container<TextDocumentContentChangeEvent>(
					new TextDocumentContentChangeEvent { Range = range, Text = text })
			}, CancellationToken.None);
		Assert.That(textDocumentHandler.Document.Get(URI), Is.EqualTo(expected));
	}

	[TestCaseSource(nameof(MultiLineTextDocumentChanges))]
	public async Task HandleMultiLineChangeTextDocumentAsync(Range range, string text, string[] expected)
	{
		await textDocumentHandler.Handle(
			new DidChangeTextDocumentParams
			{
				TextDocument = new OptionalVersionedTextDocumentIdentifier { Uri = MultiLineURI },
				ContentChanges = new Container<TextDocumentContentChangeEvent>(
					new TextDocumentContentChangeEvent { Range = range, Text = text })
			}, CancellationToken.None);
		Assert.That(textDocumentHandler.Document.Get(MultiLineURI), Is.EqualTo(expected));
	}

	[Test]
	public async Task HandleSaveTextDocumentAsync()
	{
		await textDocumentHandler.Handle(
			new DidSaveTextDocumentParams
			{
				TextDocument = new OptionalVersionedTextDocumentIdentifier { Uri = MultiLineURI }
			}, CancellationToken.None);
		var subPackage = TestPackage.Instance.FindSubPackage(MultiLineURI.Path.GetFolderName());
		Assert.That(subPackage, Is.Not.Null);
		Assert.That(subPackage?.GetType(MultiLineURI.Path.GetFileName()), Is.Not.Null);
	}

	[TestCase("has number\r\nAdd(num Number) Number\r\n\tnum + number")]
	public async Task HandleOpenTextDocumentAsync(string text)
	{
		await textDocumentHandler.Handle(
			new DidOpenTextDocumentParams
			{
				TextDocument = new TextDocumentItem { Uri = MultiLineURI, Text = text }
			}, CancellationToken.None);
		var subPackage = TestPackage.Instance.FindSubPackage(MultiLineURI.Path.GetFolderName());
		Assert.That(subPackage, Is.Not.Null);
		Assert.That(subPackage?.GetType(MultiLineURI.Path.GetFileName()), Is.Not.Null);
	}
}