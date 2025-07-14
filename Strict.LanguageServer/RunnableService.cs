using Strict.Runtime;

namespace Strict.LanguageServer;

public interface RunnableService
{
	public void Run(VirtualMachine vm);
}