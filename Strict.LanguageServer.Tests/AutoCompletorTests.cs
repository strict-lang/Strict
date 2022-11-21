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
	[TestCase(0, "Write", "has log", "Log(message Text)", "\tlog.")]
	[TestCase(2, "Start", "has range Range", "Bla", "\trange.")]
	public async Task HandleLogAutoCompleteAsync(int itemIndex, string completionName, params string[] code)
	{
		var documentUri = GetDocumentUri(completionName);
		var strictDocument = new StrictDocument();
		strictDocument.AddOrUpdate(documentUri, code);
		var autocompleteHandler = new AutoCompletor(strictDocument,
			await new PackageSetup().GetPackageAsync(Repositories.DevelopmentFolder + ".Base"));
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
}