# Bugs

- GC race: a young collection inside a pool job (more than 2 threads) frees or misses an object a job thread still uses. Repro: the -O3 seed compiler (`Compiler\Bootstrap.cmd` with -O3) building Strict.strict after strict.exe stamped the .strictbinary files, with STRICT_GC_LOG=1, crashes 4 of 4 right after a "young, pool job" collection (AccessViolation at changing places: KnownTypes.Index, TypeEntry.ShortName, CallTarget.Holding). STRICT_THREADS=1 or 2, STRICT_YOUNG=0, -O2 or no vectorizer hide it. Bootstrap.cmd links with -O2 until it is fixed; release builds use -O3.
