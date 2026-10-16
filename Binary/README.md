# Strict.Binary
BinaryData reads little endian file formats (meshes, levels, animations): BinaryData.Read loads a file as one number per byte, Int8/UInt16/Int16/Int32/UInt32/Float32 read at a byte offset and Text reads a 7 bit length prefixed string (TextEnd is the offset after it). Offsets are explicit, there is no read cursor, so parsers stay pure functions.
