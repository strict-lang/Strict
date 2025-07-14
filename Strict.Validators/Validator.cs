namespace Strict.Validators;

/// <summary>
/// All validators are run only after code is changed, loading existing code does not run them.
/// Validators can be slower than parsing and execution and do some deeper checks (ala NDepend).
/// </summary>
public interface Validator
{
	void Validate();
}