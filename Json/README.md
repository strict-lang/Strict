# Strict.Json
Json keeps the text of a value (or a whole file, `Json.Read(path)`) and reads it on demand: `json(key)` (its `get(key)`) is the value of an object member (an empty Json when missing), `Keys` the member names, `Elements` the array elements (or the member values of an object), `Values` the elements as numbers, `to Text` a string with its escapes decoded (`\n`, `\"`, `\u00e9` and surrogate pairs like `\ud83d\ude00` as UTF-8 (`JsonText`), other values as written: `true`, `null`, `82`) and `to Number` a number.

```
let skeleton = Json.Read("Json/BatSkeleton.json")
skeleton("skeleton").get("width") to Number is 205.28
skeleton("bones").Elements(2).get("parent") to Text is "Bat_Bone"
```

Every call scans the text again, so keep the Json of a nested value instead of asking the root repeatedly. The Strict VM runs `get`, `Keys`, `Elements` and `Values` as compiled code (Runtime/HostedJson), a content file reads as fast as natively. Nesting is limited by the stack. `BatSkeleton.json` is the Spine skeleton of the TinyBat sample of DeltaEngine.

JsonWriter writes JSON: start with `JsonWriter.Object` or `JsonWriter.Array` and add members with `With(name, value)` or elements with `With(value)`, the value a Text, Number, Boolean (objects only) or another JsonWriter; `text` is the result. Texts are quoted with every control character escaped (`\t`, `\n`, `\u0001`), other characters stay as they are.

```
JsonWriter.Object.With("symbol", "BTCUSDT").With("free", 1.5).With("prices", JsonWriter.Array.With(1).With(2)).text
```
