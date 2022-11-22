using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Strict.Language;

namespace Strict.LanguageServer.Tests;

public sealed class AutoCompletorTests : LanguageServerTests
{
	[SetUp]
	public async Task CreateStrictDocumentAsync()
	{
		strictDocument = new StrictDocument();
		package = await new PackageSetup().GetPackageAsync(Repositories.DevelopmentFolder + ".Base");
	}

	private StrictDocument strictDocument = null!;
	private Package package = null!;

	[TestCase(0, "Write", "has log", "Log(message Text)", "\tlog.")]
	[TestCase(2, "Start", "has range Range", "Bla", "\trange.")]
	[TestCase(0, "+", "has log", "CheckText(message Text)", "\tmessage.")]
	[TestCase(0, "+", "has log", "FirstMethod(message Text)", "\tmessage.", "SecondMethod Number", "\t5")]
	public async Task HandleLogAutoCompleteAsync(int itemIndex, string completionName, params string[] code)
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
					Position = new Position { Character = 8, Line = 2 }
				}, CancellationToken.None)).Items.ToArray()[itemIndex].Label, Is.EqualTo(completionName));
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