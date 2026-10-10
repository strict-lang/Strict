using MediatR;
using Moq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

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
		var uri = new DocumentUri("", "", "BaseTypesTest/BaseTypesTest" + Type.Extension, "", "");
		document.AddOrUpdate(uri, "has logger", "Run",
			"\tconstant worldHelper = MissingSibling(\"World\")", "\tlogger.Log(worldHelper)");
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
		var uri = new DocumentUri("", "", "HasTests/HasTests" + Type.Extension, "", "");
		document.AddOrUpdate(uri, "has number", "Run Number", "\t5 is 5", "\tnumber");
		document.InitializeContent(uri);
		var diagnostics = document.GetDiagnostics(TestPackage.Instance, uri, languageServer.Object);
		Assert.That(diagnostics.Select(item => item.Message), Is.Empty);
		var parsed = TestPackage.Instance.Find("HasTests")?.FindDirectType("HasTests")?.Methods.
			Single(method => method.Name == Method.Run);
		Assert.That(parsed?.WasParsedAlready, Is.True);
	}

	[Test]
	public void UnusedParameterIsReportedAsDiagnostic()
	{
		var uri = new DocumentUri("", "", "UnusedOther/UnusedOther" + Type.Extension, "", "");
		document.AddOrUpdate(uri, "has number", "Twice(other Number) Number", "\tTwice(2) is 0",
			"\tnumber * 2");
		document.InitializeContent(uri);
		Assert.That(document.GetDiagnostics(TestPackage.Instance, uri, languageServer.Object).
			Select(item => item.Code?.String), Has.One.EqualTo("UnusedMethodParameterMustBeRemoved"));
	}

	[Test]
	public void ToLocalFileAcceptsVsCodeEncodedWindowsUri()
	{
		var path = Path.Combine(Path.GetTempPath(), "StrictUri" + Guid.NewGuid().ToString("N"),
			"BaseTypesTest" + Type.Extension);
		var encoded = "file:///" + path.Replace('\\', '/').Replace(":", "%3A");
		Assert.That(DocumentUri.From(encoded).ToLocalFile(), Is.EqualTo(path).IgnoreCase);
	}

	[Test]
	public async Task ManualRunLoadsSiblingTypeFromTheSameFolderAsync()
	{
		var folder = Path.Combine(Path.GetTempPath(), "StrictSiblings" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(folder);
		try
		{
			await File.WriteAllTextAsync(Path.Combine(folder, "TextHelper" + Type.Extension),
				"has value Text\nGreet Text\n\t\"Hello, \" + value + \"!\"");
			var runPath = Path.Combine(folder, "BaseTypesTest" + Type.Extension);
			await File.WriteAllTextAsync(runPath, "has number\nRun Text\n\tTextHelper(\"World\").Greet");
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