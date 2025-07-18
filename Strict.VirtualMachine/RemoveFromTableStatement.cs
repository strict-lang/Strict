﻿namespace Strict.Runtime;

public class RemoveFromTableStatement(Register key, string identifier) : RegisterStatement(key,
	Instruction.RemoveFromTable)
{
	public string Identifier { get; } = identifier;
}