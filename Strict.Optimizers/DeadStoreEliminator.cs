using Strict.Runtime.Statements;

namespace Strict.Optimizers;

/// <summary>
/// Removes stores to variables that are never loaded again. A StoreVariableStatement is dead
/// if the variable identifier never appears in any LoadVariableToRegister or
/// StoreFromRegisterStatement in the rest of the instruction list. Member variables are always
/// kept as they may be accessed externally.
/// </summary>
public sealed class DeadStoreEliminator : StatementOptimizer
{
	public override List<Statement> Optimize(List<Statement> statements)
	{
		var usedVariables = CollectUsedVariables(statements);
		statements.RemoveAll(statement =>
			statement is StoreVariableStatement store && !store.IsMember &&
			!usedVariables.Contains(store.Identifier));
		return statements;
	}

	private static HashSet<string> CollectUsedVariables(List<Statement> statements)
	{
		var used = new HashSet<string>();
		foreach (var statement in statements)
		{
			if (statement is LoadVariableToRegister load)
				used.Add(load.Identifier);
			else if (statement is StoreFromRegisterStatement store)
				used.Add(store.Identifier);
		}
		return used;
	}
}