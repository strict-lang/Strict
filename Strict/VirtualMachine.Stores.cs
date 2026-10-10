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
			var value = CloneConstantValue(storeVariable.ValueInstance);
			StoreIdentifierValue(storeVariable.Identifier, value, storeVariable.IsMember);
		}
		else if (instruction.InstructionType == InstructionType.StoreRegisterToVariable)
		{
			var storeFromRegister = (StoreFromRegisterInstruction)instruction;
			if (!TryStoreToListElement(storeFromRegister))
				StoreIdentifierValue(storeFromRegister.Identifier,
					Memory.Registers[storeFromRegister.Register], false);
		}
	}

	private void TryLoadInstructions(Instruction instruction)
	{
		if (instruction.InstructionType == InstructionType.LoadVariableToRegister)
		{
			var loadVariable = (LoadVariableToRegister)instruction;
			if (!GetIdentifierAccessPath(loadVariable.Identifier).TryResolve(this, out var registerValue))
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

	private IdentifierAccessPath GetIdentifierAccessPath(string identifier) =>
		identifierAccessPaths.TryGetValue(identifier, out var accessPath)
			? accessPath
			: identifierAccessPaths[identifier] = IdentifierAccessPath.Parse(identifier);

	private bool TryGetFrameValue(int symbolId, out ValueInstance value) =>
		Memory.Frame.TryGet(symbolId, out value);

	private void StoreIdentifierValue(string identifier, ValueInstance value, bool isMember)
	{
		var accessPath = GetIdentifierAccessPath(identifier);
		if (accessPath.MemberNames.Length == 0)
		{
			Memory.Frame.Set(accessPath.RootSymbolId, value, isMember, identifier);
			return;
		}
		if (!accessPath.GetParentPath().TryResolve(this, out var parentValue))
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

	private bool TryStoreToListElement(StoreFromRegisterInstruction store)
	{
		var indexedAccessPath = GetIndexedElementAccessPath(store.Identifier);
		if (!indexedAccessPath.IsValid)
			return false;
		var listValue = TryResolveListValue(indexedAccessPath.ListPath);
		if (!listValue.IsList)
			return false;
		var indexInstance = TryResolveIndexValue(indexedAccessPath.IndexExpression);
		if (!indexInstance.HasValue)
			return false;
		var index = (int)indexInstance.Number;
		if (index >= 0 && index < listValue.List.Count)
		{
			listValue.List[index] = Memory.Registers[store.Register];
			return true;
		}
		return false;
	}

	private IndexedElementAccessPath GetIndexedElementAccessPath(string identifier) =>
		indexedElementAccessPaths.TryGetValue(identifier, out var accessPath)
			? accessPath
			: indexedElementAccessPaths[identifier] = IndexedElementAccessPath.Parse(identifier);

	private ValueInstance TryResolveListValue(string listPath) =>
		GetIdentifierAccessPath(listPath).TryResolve(this, out var listValue)
			? listValue
			: default;

	private ValueInstance TryResolveIndexValue(string indexExpression)
	{
		if (double.TryParse(indexExpression, out var number))
			return new ValueInstance(executable.numberType, number);
		var accessPath = GetIdentifierAccessPath(indexExpression);
		if (accessPath.TryResolve(this, out var indexInstance))
			return indexInstance;
		return TryGetFrameValue(IndexSymbolId, out indexInstance)
			? indexInstance
			: default;
	}

	private readonly record struct IdentifierAccessPath(int RootSymbolId, string[] MemberNames)
	{
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
				return default;
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

		public IdentifierAccessPath GetParentPath() =>
			MemberNames.Length == 1
				? this with { MemberNames = [] }
				: this with { MemberNames = MemberNames[..^1] };
	}

	private readonly record struct IndexedElementAccessPath(string ListPath,
		string IndexExpression,
		bool IsValid)
	{
		public static IndexedElementAccessPath Parse(string identifier)
		{
			var openParen = identifier.LastIndexOf('(');
			return openParen <= 0 || !identifier.EndsWith(')')
				? new IndexedElementAccessPath(string.Empty, string.Empty, false)
				: new IndexedElementAccessPath(identifier[..openParen], identifier[(openParen + 1)..^1],
					true);
		}
	}
}
