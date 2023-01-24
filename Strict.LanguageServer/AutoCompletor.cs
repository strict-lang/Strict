using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

public sealed class AutoCompletor : ICompletionHandler
{
	private readonly StrictDocument documentManager;
	private readonly Package package;

	public AutoCompletor(StrictDocument documentManager, Package package)
	{
		this.package = package;
		this.documentManager = documentManager;
	}

	public async Task<CompletionList> Handle(CompletionParams request,
		CancellationToken cancellationToken)
	{
		if (request.Context?.TriggerCharacter != "." && request.Context?.TriggerCharacter != "/n")
			return await Task.FromResult(new CompletionList()).ConfigureAwait(false);
		var code = documentManager.Get(request.TextDocument.Uri);
		var typeName = request.TextDocument.Uri.Path.GetFileName();
		var completionType = GetCompletionType(request, typeName, code);
		return completionType != null
			? await GetCompletionMethodsAsync(completionType).ConfigureAwait(false)
			: await Task.FromResult(new CompletionList()).ConfigureAwait(false);
	}

	private Type? GetCompletionType(TextDocumentPositionParams request, string typeName,
		IReadOnlyList<string> code)
	{
		var type = package.SynchronizeAndGetType(typeName, code);
		var typeToFind = GetTypeToFind(code[request.Position.Line]);
		return FindMemberTypeName(type, typeToFind) ?? FindMethod(request, code, type)?.Parameters.
				FirstOrDefault(p => p.Name == typeToFind)?.Type ??
			FindVariable(FindLine(code, typeToFind, request.Position.Line - 1), type);
	}

	private static string GetTypeToFind(string line)
	{
		var splitText = line.Split('.')[0].Split(' ', StringSplitOptions.TrimEntries);
		return splitText.Length == 1
			? splitText[0]
			: splitText[^1];
	}

	private static Type? FindMemberTypeName(Type type, string typeToFind) => type.Members.FirstOrDefault(m => m.Name == typeToFind)?.Type;

	private static Method? FindMethod(TextDocumentPositionParams request,
		IReadOnlyList<string> code, Type type) =>
		type.Methods.Count == 1
			? type.Methods[0]
			: FindMethodFromLine(request, code, type);

	private static Method? FindMethodFromLine(TextDocumentPositionParams request, IReadOnlyList<string> code, Type type)
	{
		var methodName = FindMethodName(code, request.Position.Line - 1);
		return type.Methods.FirstOrDefault(method => method.Name == methodName);
	}

	private static string? FindMethodName(IReadOnlyList<string> code, int lineNumber)
	{
		while (lineNumber > 0)
		{
			var currentLine = code[lineNumber];
			if (currentLine[0] != ' ' && currentLine[0] != '\t')
				return currentLine.Contains('(')
					? currentLine.Split('(')[0]
					: currentLine.Split(' ')[0];
			lineNumber--; //ncrunch: no coverage start
		}
		return null; //ncrunch: no coverage end
	}

	private static Type? FindVariable(string? line, Type type)
	{
		try
		{
			var expression = type.ParseExpression(line);
			return expression.ReturnType;
		}
		catch
		{
			return null;
		}
	}

	private static string? FindLine(IReadOnlyList<string> code, string variableName, int lineNumber)
	{
		while (lineNumber > 0)
		{
			var currentLine = code[lineNumber];
			if (currentLine.Contains(Constant + " " + variableName + " = "))
				return currentLine.Trim()[(Constant.Length + 1 + variableName.Length + 3)..];
			lineNumber--;
		}
		return null;
	}

	private const string Constant = "constant";

	private static Task<CompletionList> GetCompletionMethodsAsync(Type completionType) =>
		Task.FromResult(new CompletionList(
			CreateCompletionItems(GetCompletionItemsForMembersAndMethods(completionType))));

	private static List<StrictCompletionItem> GetCompletionItemsForMembersAndMethods(Type completionType)
	{
		var completionItems = CreateCompletionItemsForMethods(completionType.Methods);
		completionItems.AddRange(CreateCompletionItemsForMembers(completionType.Members));
		return completionItems;
	}

	private static List<StrictCompletionItem>
		CreateCompletionItemsForMethods(IEnumerable<Method> methods) =>
		methods.Select(method => method.Name).Where(name => name.IsWord()).
			Select(name => new StrictCompletionItem(name, CompletionItemKind.Method)).ToList();

	private static IEnumerable<StrictCompletionItem>
		CreateCompletionItemsForMembers(IEnumerable<Member> members) =>
		members.Where(member => member.IsPublic).Select(member => member.Name).Select(name =>
			new StrictCompletionItem(name, CompletionItemKind.Method));

	private static IEnumerable<CompletionItem> CreateCompletionItems(IEnumerable<StrictCompletionItem> completionItems) =>
		completionItems.Select(item => new CompletionItem
		{
			InsertText = item.Name, FilterText = item.Name, Label = item.Name, Kind = item.CompletionKind
		});

	//ncrunch: no coverage start
	public CompletionRegistrationOptions GetRegistrationOptions(CompletionCapability capability,
		ClientCapabilities clientCapabilities) =>
		new()
		{
			TriggerCharacters = new Container<string>("."),
			DocumentSelector = BaseSelectors.StrictDocumentSelector,
			ResolveProvider = false
		};
	//ncrunch: no coverage end

	private sealed record StrictCompletionItem(string Name, CompletionItemKind CompletionKind);
}