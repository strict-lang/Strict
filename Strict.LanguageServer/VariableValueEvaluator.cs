﻿using OmniSharp.Extensions.LanguageServer.Protocol.Server;

namespace Strict.LanguageServer;

public sealed class VariableValueEvaluator(ILanguageServerFacade languageServer, string[] lines)
	: RunnerService, RunnableService
{
	private const string NotificationName = "valueEvaluationNotification";

	public void Run(VirtualMachine.VirtualMachine vm)
	{
		var lineValuePair = new Dictionary<int, string>();
		for (var i = 0; i < lines.Length; i++)
			foreach (var variable in vm.Memory.Variables.Where(variable =>
				//ncrunch: no coverage start, TODO: missing tests
				lines[i].Contains(variable.Key)))
				lineValuePair[i] =
					variable.Value.Value.ToString() ?? throw new InvalidOperationException();
		//ncrunch: no coverage end
		languageServer.SendNotification(NotificationName,
			new VariableStateNotificationMessage(lineValuePair));
	}
}