using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class VirtualMachine
{
	private void TryStoreInstructions(Instruction instruction)
	{
		if (instruction.InstructionType == InstructionType.Set)
		{
			var set = (SetInstruction)instruction;
			Memory.Registers[set.Register] = CloneConstantValue(set.ValueInstance);
		}
		else if (instruction.InstructionType == InstructionType.StoreConstantToVariable)
		{
			var storeVariable = (StoreVariableInstruction)instruction;
			StoreValue(storeVariable, storeVariable.Identifier,
				CloneConstantValue(storeVariable.ValueInstance), storeVariable.IsMember);
		}
		else if (instruction.InstructionType == InstructionType.StoreRegisterToVariable)
		{
			var storeFromRegister = (StoreFromRegisterInstruction)instruction;
			StoreValue(storeFromRegister, storeFromRegister.Identifier,
				Memory.Registers[storeFromRegister.Register], false);
		}
	}

	/// <summary>
	/// Constants and registers both store into list elements like numbers(index) = 5.
	/// </summary>
	private void StoreValue(Instruction instruction, string identifier, ValueInstance value,
		bool isMember)
	{
		var storePath = instruction.CachedAccessPath ??=
			(object?)IndexedElementAccessPath.TryParse(identifier) ??
			IdentifierAccessPath.Parse(identifier);
		if (storePath is not IndexedElementAccessPath indexedPath ||
			!TryStoreToListElement(indexedPath, value))
			StoreIdentifierValue(storePath as IdentifierAccessPath ??
				((IndexedElementAccessPath)storePath).WholePath, identifier, value, isMember);
	}

	private void TryLoadInstructions(Instruction instruction)
	{
		if (instruction.InstructionType == InstructionType.LoadVariableToRegister)
		{
			var loadVariable = (LoadVariableToRegister)instruction;
			if (!GetIdentifierAccessPath(loadVariable, loadVariable.Identifier).
				TryResolve(this, out var registerValue))
				throw Fail("Could not resolve variable '" + loadVariable.Identifier + //ncrunch: no coverage
					"' - check that the variable is defined and in scope");
			Memory.Registers[loadVariable.Register] = registerValue;
		}
		else if (instruction.InstructionType == InstructionType.LoadConstantToRegister)
		{
			var loadConstant = (LoadConstantInstruction)instruction;
			Memory.Registers[loadConstant.Register] = CloneConstantValue(loadConstant.Constant);
		}
	}

	private static ValueInstance CloneConstantValue(ValueInstance value) =>
		value.IsList
			? new ValueInstance(value.List.Clone(value.List.ReturnType))
			: value.IsDictionary
				? new ValueInstance(value.GetType(),
					new Dictionary<ValueInstance, ValueInstance>(value.GetDictionaryItems()))
				: value;

	private static IdentifierAccessPath GetIdentifierAccessPath(Instruction instruction,
		string identifier) =>
		(IdentifierAccessPath)(instruction.CachedAccessPath ??= IdentifierAccessPath.Parse(identifier));

	private bool TryGetFrameValue(int symbolId, out ValueInstance value) =>
		Memory.Frame.TryGet(symbolId, out value);

	private void StoreIdentifierValue(IdentifierAccessPath accessPath, string identifier,
		ValueInstance value, bool isMember)
	{
		if (accessPath.MemberNames!.Length == 0)
		{
			Memory.Frame.Set(accessPath.RootSymbolId, value, isMember, identifier);
			return;
		}
		if (!accessPath.ParentPath.TryResolve(this, out var parentValue))
			throw Fail("Could not resolve parent path for '" + identifier + "'");
		var memberName = accessPath.MemberNames[^1];
		var flatInstance = parentValue.TryGetFlatNumericArrayInstance();
		if (flatInstance != null)
		{
			if (!flatInstance.TrySetMember(memberName, value))
				throw Fail("Could not assign member '" + identifier + "' on flat numeric array");
			return;
		}
		if (parentValue.TryGetValueTypeInstance() is not { } typeInstance)
			throw Fail("Cannot assign member '" + identifier + "' - parent is not a type instance (" +
				parentValue.GetType().Name + ")");
		if (!typeInstance.TrySetValue(memberName, value))
			throw Fail("Could not assign member '" + identifier + "' on " + typeInstance.ReturnType.Name);
	}

	private ValueInstance TryGetNativeMemberValue(ValueInstance current, string memberName) =>
		current.IsText && memberName is "characters" or Type.ElementsLowercase
			? current
			: memberName is "Length"
				? current.IsText
					? new ValueInstance(executable.numberType, current.Text.Length)
					: current.IsList
						? new ValueInstance(executable.numberType, current.List.Count)
						: default
				: default;

	private bool TryStoreToListElement(IndexedElementAccessPath indexedAccessPath,
		ValueInstance value)
	{
		var listValue = indexedAccessPath.ListPath.TryResolve(this, out var resolvedList)
			? resolvedList
			: default;
		if (!listValue.IsList)
			return false;
		var indexInstance = TryResolveIndexValue(indexedAccessPath);
		if (!indexInstance.HasValue)
			return false;
		var index = (int)indexInstance.GetArithmeticNumber();
		if (index >= 0 && index < listValue.List.Count)
		{
			listValue.List[index] = value;
			return true;
		}
		return false;
	}

	private ValueInstance TryResolveIndexValue(IndexedElementAccessPath indexedAccessPath)
	{
		if (indexedAccessPath.IndexNumber is { } number)
			return new ValueInstance(executable.numberType, number);
		if (indexedAccessPath.IndexPath.TryResolve(this, out var indexInstance))
			return indexInstance;
		return TryGetFrameValue(IndexSymbolId, out indexInstance)
			? indexInstance
			: default;
	}

	private sealed class IdentifierAccessPath(int rootSymbolId, string[]? memberNames)
	{
		private static readonly IdentifierAccessPath Unresolvable = new(-1, null);
		public int RootSymbolId { get; } = rootSymbolId;
		public string[]? MemberNames { get; } = memberNames;

		public bool TryResolve(VirtualMachine vm, out ValueInstance value)
		{
			if (MemberNames == null)
			{
				value = default;
				return false;
			}
			if (!vm.TryGetFrameValue(RootSymbolId, out var current))
			{
				value = default;
				return false;
			}
			for (var memberIndex = 0; memberIndex < MemberNames.Length; memberIndex++)
			{
				var memberName = MemberNames[memberIndex];
				if (RootSymbolId == OuterSymbolId && memberName == Type.ValueLowercase)
					continue;
				if (RootSymbolId == OuterSymbolId && memberName == Type.IndexLowercase &&
					vm.TryGetFrameValue(OuterIndexSymbolId, out var outerIndexValue))
				{
					current = outerIndexValue;
					continue;
				}
				var nativeMemberValue = vm.TryGetNativeMemberValue(current, memberName);
				if (nativeMemberValue.HasValue)
				{
					current = nativeMemberValue;
					continue;
				}
				if (current.TryGetFlatNumericMember(memberName, out var flatMember))
				{
					current = flatMember;
					continue;
				}
				var typeInstance = current.TryGetValueTypeInstance();
				if (typeInstance == null || !typeInstance.TryGetValue(memberName, out current))
				{
					value = default;
					return false;
				}
			}
			value = current;
			return true;
		}

		public static IdentifierAccessPath Parse(string identifier)
		{
			if (identifier == Type.None)
				return Unresolvable;
			var firstDotIndex = identifier.IndexOf('.');
			if (firstDotIndex < 0)
				return new IdentifierAccessPath(CallFrame.ResolveSymbolId(identifier), []);
			var rootSymbolId = CallFrame.ResolveSymbolId(identifier[..firstDotIndex]);
			var memberCount = 1;
			for (var index = firstDotIndex + 1; index < identifier.Length; index++)
				if (identifier[index] == '.')
					memberCount++;
			var memberNames = new string[memberCount];
			var memberIndex = 0;
			var segmentStart = firstDotIndex + 1;
			while (segmentStart < identifier.Length)
			{
				var nextDotIndex = identifier.IndexOf('.', segmentStart);
				memberNames[memberIndex++] = nextDotIndex < 0
					? identifier[segmentStart..]
					: identifier[segmentStart..nextDotIndex];
				if (nextDotIndex < 0)
					break;
				segmentStart = nextDotIndex + 1;
			}
			return new IdentifierAccessPath(rootSymbolId, memberNames);
		}

		public IdentifierAccessPath ParentPath =>
			field ??= new IdentifierAccessPath(RootSymbolId, MemberNames![..^1]);
	}

	private sealed class IndexedElementAccessPath(string identifier, int openParen)
	{
		public IdentifierAccessPath WholePath { get; } = IdentifierAccessPath.Parse(identifier);
		public IdentifierAccessPath ListPath { get; } =
			IdentifierAccessPath.Parse(identifier[..openParen]);
		public double? IndexNumber { get; } =
			double.TryParse(identifier.AsSpan(openParen + 1, identifier.Length - openParen - 2),
				out var number)
				? number
				: null;
		public IdentifierAccessPath IndexPath { get; } =
			IdentifierAccessPath.Parse(identifier[(openParen + 1)..^1]);

		public static IndexedElementAccessPath? TryParse(string identifier)
		{
			var openParen = identifier.LastIndexOf('(');
			return openParen <= 0 || !identifier.EndsWith(')')
				? null
				: new IndexedElementAccessPath(identifier, openParen);
		}
	}
}
