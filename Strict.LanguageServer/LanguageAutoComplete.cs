using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

public sealed class LanguageAutoComplete : ICompletionHandler
{
	private readonly StrictDocument documentManager;
	private Package? package;
	private readonly PackageSetup packageSetup;

	public LanguageAutoComplete(StrictDocument documentManager, PackageSetup packageSetup)
	{
		this.packageSetup = packageSetup;
		this.documentManager = documentManager;
	}

	public async Task<CompletionList> Handle(CompletionParams request,
		CancellationToken cancellationToken)
	{
		if (request.Context?.TriggerCharacter != ".")
			return await Task.FromResult(new CompletionList()).ConfigureAwait(false);
		var code = documentManager.Get(request.TextDocument.Uri);
		var typeName = request.TextDocument.Uri.Path.GetFileName();
		var member = await GetMemberAsync(request, typeName, code);
		if (member != null)
			return await GetCompletionMethodsAsync(member.Type.Name).ConfigureAwait(false);
		return await Task.FromResult(new CompletionList()).ConfigureAwait(false);
	}

	private async Task<Member?> GetMemberAsync(TextDocumentPositionParams request, string typeName,
		IReadOnlyList<string> code)
	{
		package ??= await packageSetup.GetPackageAsync(Repositories.DevelopmentFolder + ".Base");
		var type = package.FindDirectType(typeName) ?? new Type(package,
				new TypeLines(typeName, code.Select(line => line.Replace("    ", "\t")).ToArray())).
			ParseMembersAndMethods(new MethodExpressionParser());
		var typeToFind = code[request.Position.Line].Split('.')[0].Trim();
		var member = type.Members.FirstOrDefault(member => member.Name == typeToFind);
		return member;
	}

	private async Task<CompletionList> GetCompletionMethodsAsync(string typeName)
	{
		var completionType = package?.FindType(typeName);
		if (completionType != null)
			return await Task.FromResult(new CompletionList(
					CreateCompletionItems(completionType.Methods.Select(method => method.Name)))).
				ConfigureAwait(false);
		return await Task.FromResult(new CompletionList()).ConfigureAwait(false);
	}

	private static IEnumerable<CompletionItem> CreateCompletionItems(IEnumerable<string> items) =>
		items.Select(item => new CompletionItem
		{
			InsertText = item, FilterText = item, Label = item, Kind = CompletionItemKind.Method
		});

	public CompletionRegistrationOptions GetRegistrationOptions(CompletionCapability capability,
		ClientCapabilities clientCapabilities) =>
		new()
		{
			TriggerCharacters = new Container<string>("."),
			DocumentSelector = BaseSelectors.StrictDocumentSelector,
			ResolveProvider = true
		};
}