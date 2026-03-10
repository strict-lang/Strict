using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using Strict.Language;

namespace Strict.LanguageServer;

public sealed class VariableValueEvaluator(Package package, ILanguageServerFacade languageServer, string[] lines)
	: RunnerService(package), RunnableService
{
	private const string NotificationName = "valueEvaluationNotification";

	public void Run(VirtualMachine vm)
	{
		var lineValuePair = new Dictionary<int, string>();
		for (var i = 0; i < lines.Length; i++)
			foreach (var variable in vm.Memory.Variables.Where(variable =>
				//ncrunch: no coverage start
				lines[i].Contains(variable.Key)))
				lineValuePair[i] = variable.Value.IsText
					? variable.Value.Text
					: variable.Value.Number.ToString();
		//ncrunch: no coverage end
		languageServer.SendNotification(NotificationName,
			new VariableStateNotificationMessage(lineValuePair));
	}
}