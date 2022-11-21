using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace Strict.LanguageServer.Tests;

public sealed class TextDocumentSynchronizerTests : LanguageServerTests
{
	private static readonly DocumentUri MultiLineURI = new("", "", "Test/MultiLine.strict", "", "");

	[SetUp]
	public void MultiLineSetup() =>
		textDocumentHandler.Document.AddOrUpdate(MultiLineURI,
			"has number",
			"Add(num Number) Number",
			"\tnum + number");

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
		}
		//ncrunch: no coverage end
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
			}, new CancellationToken());
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
			}, new CancellationToken());
		Assert.That(textDocumentHandler.Document.Get(MultiLineURI), Is.EqualTo(expected));
	}
}