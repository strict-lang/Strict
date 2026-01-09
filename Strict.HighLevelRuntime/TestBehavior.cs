namespace Strict.HighLevelRuntime;

public enum TestBehavior
{
	/// <summary>
	/// Default behavior, whenever any method is first executed, all its tests are run automatically
	/// to make sure the method works as expected. This can't be disabled, but we optimize this away
	/// and cache the result till the next change is detected. Tests always have to be very fast!
	/// </summary>
	OnFirstRun,
	/// <summary>
	/// While the normal behavior OnFirstRun is great for normal execution, while developing not all
	/// methods are called all the time. So while changing things around, it is useful to force
	/// execution of all tests in a package, type, or specific method. Always used from
	/// TestRunner.TestExecutor, it will only run test code. Normal method code is not executed!
	/// </summary>
	TestRunner,
	/// <summary>
	/// Completely disable all test execution, all comparison expressions will be skipped. Only used
	/// for tests to focus on functionality and performance when we know the tests are ok anyway.
	/// </summary>
	Disabled
}