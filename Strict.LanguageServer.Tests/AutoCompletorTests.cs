using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Strict.Language;
using Strict.Expressions;

namespace Strict.LanguageServer.Tests;

public sealed class AutoCompletorTests : LanguageServerTests
{
	[SetUp]
	public async Task CreateStrictDocumentAsync()
	{
		strictDocument = new StrictDocument();
		package = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
	}

	private StrictDocument strictDocument = null!;
	private Package package = null!;

	// @formatter:off
	[TestCase("Write", 2,
		"has log",
		"Log(message Text)",
		"\tlog.")]
	[TestCase("for", 2,
		"has range Range",
		"Bla",
		"\trange.")]
	[TestCase("to", 2,
		"has log",
		"CheckText(message Text)",
		"\tmessage.")]
	[TestCase("to", 4,
		"has log",
		"SecondMethod Number",
		"\t5",
		"FirstMethod(message Text)",
		"\tmessage.")]
	[TestCase("to", 3,
		"has text",
		"VariableCall",
		"\tconstant something = \"hello\" + text",
		"\tsomething.",
		"SecondMethod Number",
		"\t5")]
	[TestCase("not", 5,
		"has text",
		"UnusedMethod",
		"\tconstant result = 5",
		"TriggerInMiddleOfTheLine",
		"\tconstant result = true",
		"\tconstant another = result.")] // @formatter:on
	public async Task HandleLogAutoCompleteAsync(string completionName, int lineNumber, params string[] code)
	{
		var documentUri = GetDocumentUri(completionName == "+"
			? "Plus"
			: completionName);
		strictDocument.AddOrUpdate(documentUri, code);
		var autocompleteHandler = new AutoCompletor(strictDocument, package);
		Assert.That(
			(await autocompleteHandler.Handle(
				new CompletionParams
				{
					Context = new CompletionContext { TriggerCharacter = "." },
					TextDocument = new TextDocumentIdentifier(documentUri),
					Position = new Position { Character = 8, Line = lineNumber }
				}, CancellationToken.None)).Items.ToArray()[0].Label, Is.EqualTo(completionName));
	}

	private static DocumentUri GetDocumentUri(string seed) =>
		new("", "", $"Test{seed}.strict", "", "");

	[TestCase(2, "Write", "has log", "Log(message Text)", "\trandom.")]
	[TestCase(1, "Write", "has log", "has some Text", "Log(message Text)", "\trandom.")]
	public async Task HandleInvalidAutoCompleteAsync(int triggerLine, string completionName, params string[] code)
	{
		var documentUri = GetDocumentUri(completionName);
		strictDocument.AddOrUpdate(documentUri, code);
		var autocompleteHandler = new AutoCompletor(strictDocument, package);
		Assert.That(
			(await autocompleteHandler.Handle(
				new CompletionParams
				{
					Context = new CompletionContext { TriggerCharacter = "." },
					TextDocument = new TextDocumentIdentifier(documentUri),
					Position = new Position { Character = 8, Line = triggerLine }
				}, CancellationToken.None)).Items.Count(), Is.EqualTo(0));
	}

	[Test]
	public async Task HandleInvalidTriggerCharacterAsync() =>
		Assert.That(
			(await new AutoCompletor(strictDocument, package).Handle(
				new CompletionParams
				{
					Context = new CompletionContext { TriggerCharacter = "/" },
					TextDocument =
						new TextDocumentIdentifier(GetDocumentUri(nameof(HandleInvalidTriggerCharacterAsync))),
					Position = new Position { Character = 8, Line = 0 }
				}, CancellationToken.None)).Items.Count(), Is.EqualTo(0));
}