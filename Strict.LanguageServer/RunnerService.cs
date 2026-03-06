using Strict.Language;
using Strict.Runtime;

namespace Strict.LanguageServer;

public class RunnerService(Package package)
{
	private BytecodeInterpreter VmInstance { get; } = new(package);
	private readonly List<RunnableService> services = new();

	public RunnerService AddService(RunnableService runnableService)
	{
		services.Add(runnableService);
		return this;
	}

	public void RunAllServices()
	{
		foreach (var service in services)
			service.Run(VmInstance);
	}
}