namespace Strict.LanguageServer;

public class RunnerService
{
	public RunnerService() => VmInstance = new VirtualMachine.VirtualMachine();
	private VirtualMachine.VirtualMachine VmInstance { get; }
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