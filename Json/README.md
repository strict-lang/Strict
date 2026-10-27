# Strict.Json
Json keeps the text of a value (or a whole file, `Json.Read(path)`) and reads it on demand: `json(key)` (its `get(key)`) is the value of an object member (an empty Json when missing), `Keys` the member names, `Elements` the array elements (or the member values of an object), `Values` the elements as numbers, `to Text` a string with its escapes decoded (`\n`, `\"`, `\u00e9` and surrogate pairs like `\ud83d\ude00` as UTF-8 (`JsonText`), other values as written: `true`, `null`, `82`) and `to Number` a number.

```
let skeleton = Json.Read("Json/BatSkeleton.json")
skeleton("skeleton").get("width") to Number is 205.28
skeleton("bones").Elements(2).get("parent") to Text is "Bat_Bone"
```

Every call scans the text again, so keep the Json of a nested value instead of asking the root repeatedly. Nesting is limited by the stack. `BatSkeleton.json` is the Spine skeleton of the TinyBat sample of DeltaEngine.
