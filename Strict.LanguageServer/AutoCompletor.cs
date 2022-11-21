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
		var member = GetMember(request, typeName, code);
		if (member != null)
			return await GetCompletionMethodsAsync(member.Type.Name).ConfigureAwait(false);
		return await Task.FromResult(new CompletionList()).ConfigureAwait(false);
	}

	private Member? GetMember(TextDocumentPositionParams request, string typeName,
		IReadOnlyList<string> code)
	{
		var type = package.SynchronizeAndGetType(typeName, code);
		var typeToFind = code[request.Position.Line].Split('.')[0].Trim();
		var member = type.Members.FirstOrDefault(member => member.Name == typeToFind);
		return member;
	}

	private async Task<CompletionList> GetCompletionMethodsAsync(string typeName)
	{
		var completionType = package.FindType(typeName);
		if (completionType != null)
		{
			var completionItems = GetCompletionItemsForMembersAndMethods(completionType);
			return await Task.FromResult(new CompletionList(CreateCompletionItems(completionItems))).
				ConfigureAwait(false);
		}
		return await Task.FromResult(new CompletionList()).ConfigureAwait(false);
	}

	private static List<StrictCompletionItem> GetCompletionItemsForMembersAndMethods(Type completionType)
	{
		var completionItems = CreateCompletionItemsForMethods(completionType.Methods);
		completionItems.AddRange(CreateCompletionItemsForMembers(completionType.Members));
		return completionItems;
	}

	private static List<StrictCompletionItem>
		CreateCompletionItemsForMethods(IEnumerable<Method> methods) =>
		methods.Select(method => method.Name).
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

	public CompletionRegistrationOptions GetRegistrationOptions(CompletionCapability capability,
		ClientCapabilities clientCapabilities) =>
		new()
		{
			TriggerCharacters = new Container<string>("."),
			DocumentSelector = BaseSelectors.StrictDocumentSelector,
			ResolveProvider = true
		};

	private sealed record StrictCompletionItem(string Name, CompletionItemKind CompletionKind);
}