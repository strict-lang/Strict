using System.Diagnostics;
using System.IO.Pipelines;
using System.IO.Pipes;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Nerdbank.Streams;
using OmniSharp.Extensions.LanguageServer.Server;
using Strict.Language;
using Strict.Language.Expressions;
using Strict.LanguageServer;
using PipeOptions = System.IO.Pipes.PipeOptions;

var (input, output) = await CreateAndGetPipeline();
Logger.Info("Connecting...");
// @formatter:off
 var strictBase = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
var server = await LanguageServer.From(options =>
	options.WithInput(input)
		.WithOutput(output)
		.WithLoggerFactory(new LoggerFactory())
		.AddDefaultLoggingProvider()
		.WithServices(services => ConfigureServices(services, strictBase))
		.WithHandler<TextDocumentSynchronizer>()
		.WithHandler<AutoCompletor>()
		.WithHandler<CommandExecutor>()
		.WithHandler<DocumentHighlighter>());
Logger.Info("Client connected!");
await server.WaitForExit;
// @formatter:on
await Task.WhenAny(Task.Run(async () =>
{
	while (true)
	{
		await Task.Delay(1000);
		if (server.ClientSettings.ProcessId.HasValue &&
			Process.GetProcessById((int)server.ClientSettings.ProcessId.Value).HasExited)
		{
			Logger.Info("Client disconnected");
			server.ForcefulShutdown();
			return;
		}
	}
}), server.WaitForExit);

static void ConfigureServices(IServiceCollection services, Package strictBase)
{
	services.AddSingleton<StrictDocument>();
	services.AddSingleton(strictBase);
}

static async Task<(PipeReader input, PipeWriter output)> CreateAndGetPipeline()
{
	var pipe = new NamedPipeServerStream("Strict.LanguageServer", PipeDirection.InOut, 1,
		PipeTransmissionMode.Byte, PipeOptions.Asynchronous);
	await pipe.WaitForConnectionAsync();
	var pipeline = pipe.UsePipe();
	return (pipeline.Input, pipeline.Output);
}