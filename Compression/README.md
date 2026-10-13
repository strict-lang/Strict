# Strict.Compression
Deflate written in Strict, used for PNG images: `Inflate` decompresses zlib streams (stored blocks and fixed or dynamic Huffman codes, `Inflation` keeps the state while decoding), `Deflate` compresses with LZ77 and the fixed Huffman codes, `Huffman` and `Symbol` build and read the codes. The tests at the start of each method in `Inflate.strict` are the best place to start.
