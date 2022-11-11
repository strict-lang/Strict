using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;

namespace Strict.LanguageServer.Tests
{
	public class LanguageAutoCompleteTests : LanguageServerTests
	{
		private LanguageAutoComplete autocompleteHandler = null!;

		[SetUp]
		public void InitializeHandler()
		{
			var strictDocument = new StrictDocument();
			strictDocument.AddOrUpdate(URI, "has log", "Log(message Text)", "\tlog.");
			autocompleteHandler = new LanguageAutoComplete(strictDocument, new PackageSetup());
		}

		[Test]
		public async Task HandleLogAutoCompleteAsync() =>
			Assert.That(
				(await autocompleteHandler.Handle(
					new CompletionParams
					{
						Context = new CompletionContext { TriggerCharacter = "." },
						TextDocument = new TextDocumentIdentifier(URI),
						Position = new Position { Character = 8, Line = 2 }
					}, CancellationToken.None)).Items.ToArray()[0].Label, Is.EqualTo("Write"));
	}
}