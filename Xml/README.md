# Strict.Xml
Xml reads one line of a simple XML file at a time, enough for the content, level and effect files of the C# DeltaEngine: `Attribute(key)` returns the quoted value (empty when missing), `Values(key)` the numbers in it (`"0 0.5 0 0"` or `"(1.00, 0.50, 1.00, 1.00)"`). No nesting, entities or multi-line elements; use a native library if a format needs them.
