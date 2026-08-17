using MediatR;
using Moq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Window;
using Strict.Language;
using Strict.Language.Tests;

namespace Strict.LanguageServer.Tests;

public sealed class CommandExecutorTests
{
	[SetUp]
	public void CreateMocks()
	{
		notifications.Clear();
		var window = new Mock<IWindowLanguageServer>();
		window.Setup(expression => expression.SendNotification(It.IsAny<string>()));
		window.Setup(expression => expression.SendNotification(It.IsAny<LogMessageParams>()));
		languageServer = new Mock<ILanguageServer>();
		languageServer.Setup(expression => expression.Window).Returns(window.Object);
		languageServer.Setup(expression =>
				expression.SendNotification(It.IsAny<string>(), It.IsAny<object>())).
			Callback<string, object>((name, payload) =>
			{
				if (name == "testRunnerNotification" && payload is TestNotificationMessage message)
					notifications.Add(message);
			});
		document = new StrictDocument(TestPackage.Instance);
	}

	private readonly List<TestNotificationMessage> notifications = [];
	private Mock<ILanguageServer> languageServer = null!;
	private StrictDocument document = null!;

	[Test]
	public void OpeningRunDoesNotParseOrExecuteManualRun()
	{
		var uri = new DocumentUri("", "", "BaseTypesTest/BaseTypesTest" + Language.Type.Extension,
			"", "");
		document.AddOrUpdate(uri, "has logger", "Run",
			"\tconstant worldHelper = MissingSibling(\"World\")", "\tlogger.Log(\"hi\")");
		document.InitializeContent(uri);
		var diagnostics = document.GetDiagnostics(TestPackage.Instance, uri, languageServer.Object);
		Assert.That(diagnostics.Select(item => item.Message), Is.Empty);
		Assert.That(notifications, Is.Empty);
		var parsed = TestPackage.Instance.Find("BaseTypesTest")?.FindDirectType("BaseTypesTest")?.
			Methods.Single(method => method.Name == Method.Run);
		Assert.That(parsed?.WasParsedAlready, Is.False);
	}

	[Test]
	public void OpeningRunWithInlineTestsStillRunsThem()
	{
		var uri = new DocumentUri("", "", "HasTests/HasTests" + Language.Type.Extension, "", "");
		document.AddOrUpdate(uri, "has number", "Run", "\t5 is 5");
		document.InitializeContent(uri);
		var diagnostics = document.GetDiagnostics(TestPackage.Instance, uri, languageServer.Object);
		Assert.That(diagnostics.Select(item => item.Message), Is.Empty);
		var parsed = TestPackage.Instance.Find("HasTests")?.FindDirectType("HasTests")?.
			Methods.Single(method => method.Name == Method.Run);
		Assert.That(parsed?.WasParsedAlready, Is.True);
	}

	[Test]
	public void ToLocalFileAcceptsVsCodeEncodedWindowsUri()
	{
		var path = Path.Combine(Path.GetTempPath(), "StrictUri" + Guid.NewGuid().ToString("N"),
			"BaseTypesTest" + Language.Type.Extension);
		var encoded = "file:///" + path.Replace('\\', '/').Replace(":", "%3A");
		Assert.That(DocumentUri.From(encoded).ToLocalFile(),
			Is.EqualTo(path).IgnoreCase);
	}

	[Test]
	public async Task ManualRunLoadsSiblingTypeFromTheSameFolderAsync()
	{
		var folder = Path.Combine(Path.GetTempPath(), "StrictSiblings" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(folder);
		try
		{
			await File.WriteAllTextAsync(Path.Combine(folder, "TextHelper" + Language.Type.Extension),
				"has value Text\nGreet Text\n\t\"Hello, \" + value + \"!\"");
			var runPath = Path.Combine(folder, "BaseTypesTest" + Language.Type.Extension);
			await File.WriteAllTextAsync(runPath,
				"has number\nRun Text\n\tTextHelper(\"World\").Greet");
			var encoded = "file:///" + runPath.Replace('\\', '/').Replace(":", "%3A");
			var uri = DocumentUri.From(encoded);
			document.AddOrUpdate(uri, await File.ReadAllLinesAsync(runPath));
			var executor = new CommandExecutor(languageServer.Object, document, TestPackage.Instance);
			await ((IRequestHandler<ExecuteCommandParams, Unit>)executor).Handle(
				new ExecuteCommandParams
				{
					Command = "strict-vscode-client.run",
					Arguments = [new JObject { ["label"] = "Run" }, encoded]
				}, CancellationToken.None);
			Assert.That(notifications, Has.Count.EqualTo(1));
			Assert.That(notifications[0].State, Is.EqualTo(TestState.Green),
				notifications[0].Message + "\n" + notifications[0].StackTrace);
			Assert.That(notifications[0].MethodName, Is.EqualTo(Method.Run));
		}
		finally
		{
			Directory.Delete(folder, true);
		}
	}
}
