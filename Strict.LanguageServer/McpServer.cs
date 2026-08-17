using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Strict.Language;

namespace Strict.LanguageServer;

public static class McpServer
{
	private const string ProtocolVersion = "2024-11-05";
	private static readonly JsonSerializerOptions JsonOptions = new()
	{
		PropertyNamingPolicy = JsonNamingPolicy.CamelCase
	};

	public static async Task RunAsync(Package package, Stream? input = null, Stream? output = null,
		CancellationToken cancellationToken = default)
	{
		input ??= Console.OpenStandardInput();
		output ??= Console.OpenStandardOutput();
		using var reader = new StreamReader(input, Encoding.UTF8, false, 4096, true);
		while (!cancellationToken.IsCancellationRequested)
		{
			var body = await ReadMessageAsync(reader, cancellationToken);
			if (body == null)
				return;
			var response = Handle(body, package);
			if (response.Length == 0)
				continue;
			await WriteMessageAsync(output, response, cancellationToken);
		}
	}

	public static string Handle(string requestJson, Package package)
	{
		JsonNode? root;
		try
		{
			root = JsonNode.Parse(requestJson);
		}
		catch (JsonException)
		{
			return Error(null, -32700, "Parse error");
		}
		if (root is not JsonObject request)
			return Error(null, -32600, "Invalid request");
		var id = request["id"];
		var method = request["method"]?.GetValue<string>();
		if (method is null)
			return Error(id, -32600, "Missing method");
		if (method.StartsWith("notifications/", StringComparison.Ordinal))
			return "";
		try
		{
			return method switch
			{
				"initialize" => Result(id, new
				{
					protocolVersion = ProtocolVersion,
					capabilities = new { tools = new { } },
					serverInfo = new { name = "scrunch", version = "0.1.0" }
				}),
				"ping" => Result(id, new { }),
				"tools/list" => Result(id, new { tools = Tools }),
				"tools/call" => CallTool(id, request["params"], package),
				_ => Error(id, -32601, "Method not found: " + method)
			};
		}
		catch (Exception exception)
		{
			return Error(id, -32603, exception.Message);
		}
	}

	private static readonly object[] Tools =
	[
		new
		{
			name = "check",
			description =
				"Analyze Strict .strict files (parse, diagnostics, SCrunch tests). path is a file or folder. Uses a fresh sibling .strictbinary as a pass cache unless force is true. Returns ok plus any problems. Use this after editing .strict files instead of asking a human to look in VS Code.",
			inputSchema = new
			{
				type = "object",
				properties = new
				{
					path = new
					{
						type = "string",
						description = "File or folder to analyze. Defaults to the current directory."
					},
					force = new
					{
						type = "boolean",
						description = "Ignore .strictbinary cache and re-run parse + tests."
					}
				}
			}
		},
		new
		{
			name = "status",
			description =
				"Report .strictbinary cache freshness for a file or folder without running tests. Fresh cache means last successful parse + build + passing tests.",
			inputSchema = new
			{
				type = "object",
				properties = new
				{
					path = new
					{
						type = "string",
						description = "File or folder. Defaults to the current directory."
					}
				}
			}
		}
	];

	private static string CallTool(JsonNode? id, JsonNode? rawParams, Package package)
	{
		var name = rawParams?["name"]?.GetValue<string>();
		var arguments = rawParams?["arguments"] as JsonObject;
		var path = arguments?["path"]?.GetValue<string>();
		if (string.IsNullOrWhiteSpace(path))
			path = Environment.CurrentDirectory;
		var force = arguments?["force"]?.GetValue<bool>() ?? false;
		object payload = name switch
		{
			"check" => Summarize(ScrunchAnalyzer.AnalyzePath(package, path, force)),
			"status" => Summarize(ScrunchAnalyzer.Status(path)),
			_ => throw new InvalidOperationException("Unknown tool: " + name)
		};
		var text = JsonSerializer.Serialize(payload, JsonOptions);
		return Result(id, new { content = new[] { new { type = "text", text } }, isError = false });
	}

	private static object Summarize(FolderReport report) =>
		new
		{
			ok = report.Ok, path = report.Path, files = report.Files, cached = report.Cached,
			failed = report.Failed, testsPassed = report.TestsPassed, testsFailed = report.TestsFailed,
			problems = report.FilesReports.SelectMany(file => file.Problems.Select(problem => new
			{
				file = file.Path, line = problem.Line, kind = problem.Kind, message = problem.Message
			}))
		};

	private static string Result(JsonNode? id, object result) =>
		JsonSerializer.Serialize(new Dictionary<string, object?>
		{
			["jsonrpc"] = "2.0", ["id"] = ReadId(id), ["result"] = result
		}, JsonOptions);

	private static string Error(JsonNode? id, int code, string message) =>
		JsonSerializer.Serialize(new Dictionary<string, object?>
		{
			["jsonrpc"] = "2.0", ["id"] = ReadId(id),
			["error"] = new Dictionary<string, object> { ["code"] = code, ["message"] = message }
		}, JsonOptions);

	private static object? ReadId(JsonNode? id)
	{
		if (id is null)
			return null;
		if (id is JsonValue value && value.TryGetValue(out long number))
			return number;
		return id.GetValue<string>();
	}

	private static async Task<string?> ReadMessageAsync(StreamReader reader,
		CancellationToken cancellationToken)
	{
		var length = -1;
		while (true)
		{
			var header = await reader.ReadLineAsync(cancellationToken);
			if (header == null)
				return null;
			if (header.Length == 0)
				break;
			const string prefix = "Content-Length:";
			if (header.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) &&
				int.TryParse(header[prefix.Length..].Trim(), out var parsed))
				length = parsed;
		}
		if (length < 0)
			return null;
		var buffer = new char[length];
		var read = 0;
		while (read < length)
		{
			var n = await reader.ReadAsync(buffer.AsMemory(read, length - read), cancellationToken);
			if (n == 0)
				return null;
			read += n;
		}
		return new string(buffer);
	}

	private static async Task WriteMessageAsync(Stream output, string body,
		CancellationToken cancellationToken)
	{
		var bytes = Encoding.UTF8.GetBytes(body);
		var header = Encoding.ASCII.GetBytes("Content-Length: " + bytes.Length + "\r\n\r\n");
		await output.WriteAsync(header, cancellationToken);
		await output.WriteAsync(bytes, cancellationToken);
		await output.FlushAsync(cancellationToken);
	}
}
