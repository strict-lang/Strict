# Strict.NeuralNetworks
Neural networks written in Strict, from single neurons to large language models, and the package that drives the parallel CPU loops, SIMD and GPU kernels of the native compiler. Programs are built from the repository root (`strict build NeuralNetworks/Iris.strict`); the tests use tiny model files checked in next to them (`tiny*.gguf`, `tiny*.safetensors`), bigger models and data are downloaded or passed as paths and never committed (the MNIST digits are in [Mnist](https://github.com/strict-lang/Mnist)).

## Small networks
- `Layer` is a fully connected layer (one weight row per neuron), `Network` trains layers with backpropagation (softmax and cross entropy for several outputs, momentum), `Csv` and `Dataset` read, scale and one-hot encode samples. `Xor`, `Iris` and `Diabetes` are the first programs to run.
- `LogicGates` (`Perceptron`) and `LinearRegression` are even simpler, `Activation` gives a layer Sigmoid, Tanh, Relu, LeakyRelu, Softplus or Swish (`ActivationComparison`), `Optimizer` trains with Adam or RMSprop instead of momentum (`OptimizerComparison`).
- `Recurrent` is an Elman network trained with backpropagation through time (`BinaryAddition`), `Gru` adds update and reset gates and learns the embedded Reber grammar (`ReberPrediction`), which the Elman network can not.
- Evolution instead of gradients: `Weasel`, `Neuroevolution` (XOR weights) and `ArmEvolution`, a genetic algorithm (`Reach`) for the joint angles of an `Arm` (Denavit-Hartenberg chains like the Franka Emika Panda).

## Model files
- `Safetensors` reads `.safetensors` files (F32, F16, BF16 and I8 tensors through `TensorFile` in `native/`), `Written` writes numbers as a new file Strict, PyTorch and numpy read back, `Tuned` merges LoRAs into int8 rows at load like ComfyUI.
- `Gguf` reads llama.cpp's GGUF files through `GgufFile` (F32, F16, BF16, Q8_0, Q5_1, Q4_K, Q5_K, Q6_K, IQ4_NL, Q2_0, PQ2_0 and PTQ1_0, blocks decode to floats exactly like ggml). `Gguf(path, bits)` keeps rows as 32, 16 or 8 bit numbers; with 4 it keeps Q4_K and Q6_K blocks and the ternary PTQ1_0 blocks as they are (`strict.h` describes the packed formats), which the GPU multiplies directly.
- `Float8` (E4M3) and `BFloat16` are the number formats of this package: a GPU loop of a method returning `BFloat16s` stores bfloat16 (from 2^20 numbers on), meant for results only int8 or 16 bit tensor core kernels read.

The scratch buffers, metadata and file handles of TensorFile and GgufFile belong to the calling thread and are freed when it exits; a returned buffer stays valid until that thread's next call. Dispatch into each provider's worker pool is serialized, its chunks still run in parallel. `python NeuralNetworks/check_native_threads.py` (also with `GgufFile`) checks overlapping callers.

## GPT-2
`Gpt2Block` (layer norm, attention with a key and value cache, GELU), `Gpt2` and the byte pair tokenizer `Gpt2Tokens` run GPT-2 with 32, 16 or 8 bit weights, token for token like PyTorch with the same weights. `strict build NeuralNetworks/Gpt2Text.strict` downloads the 124M model into `gpt2/` and continues a prompt, `Gpt2Benchmark` measures decoding.

`Evaluation` scores held-out text with `Perplexity` (perplexity and bits per character), `LayerDropping` and `NeuronPruning` (FLAP with bias compensation) shrink a model without retraining, `Gpt2Training`, `Gpt2Pass` and `Finetuning` train the biases and norms (BitFit) with Adam through a backward pass over the stored 16 bit rows.

## Qwen3 and its relatives
`Qwen3.Loaded(path, bits)` loads a GGUF model and `Generation` continues a prompt greedily or with `Sampling` (temperature, top-k, top-p like llama.cpp's sampler chain). `strict build NeuralNetworks/Qwen3Benchmark.strict`, then `NeuralNetworks/Qwen3Benchmark.exe <gguf> 4 30` prints the continued text and the prefill and decode times (further arguments: prompt repeats, sequences per step, cache bits, temperature).
- **Qwen3** (dense and `qwen3moe` mixtures of experts): `Qwen3Block` (RMS norm, grouped query attention with normalized and rotated queries and keys, SwiGLU), `Experts` and `Routing` choose and run the experts of every token, `Batch` multiplies a matrix with all tokens of a prompt or of several sequences at once (`Qwen3.Copies`, `Steps`, `Nexts`), `Qwen3(model, cacheBits = 8)` keeps the keys and values as 8 bit floats. `MixturePruning` removes experts and layers.
- **Qwen3.8** (`qwen35`): every fourth layer is gated attention, the others are Gated DeltaNet layers (`DeltaBlock`, a decayed memory per value head corrected by the delta rule) grouped into a `Hybrid`; prompts go through the memory in chunks of 16 tokens (`DeltaChunks`, the chunked delta rule on the 16 bit tensor cores of `Products`).
- **Ternary Bonsai** (Prism ML's Qwen3.8 models with weights -1, 0 and 1): `Qwen3.Loaded(path, 4)` keeps the PTQ1_0 blocks (1.75 bits per weight) and `Basis` rotates the input of every folded product with `Sylvester` (normalized Sylvester-Hadamard blocks), giving the tokens of Prism's llama.cpp.
- **Qwen3.8-Flash-Next** (`qwen4exp`, 104 GB): `strict build NeuralNetworks/FlashNextText.strict`, then `NeuralNetworks/FlashNextText.exe <folder>`. `FlashLayer` mixes the hyper connection streams (`HyperStreams`), `MappedExperts` runs the chosen experts on the CPU through ggml, `Ngrams` and `NgramEmbedding` give the per layer embeddings; `FlashDrafter` and `FlashSpeculation` add multi token prediction with a separate draft head GGUF.
- `Speculation` and `LookupDecoding` verify drafts looked up in the context in one `Steps` pass, lossless for greedy decoding and much faster while a model copies text it has seen.

## GPU
On an NVIDIA GPU a whole step runs there and only the chosen tokens come back: loops become kernels, the products of packed rows take 8 bit inputs (dp4a, and the int8 tensor cores from 5 tokens on), `Attention.Merged` is a FlashAttention-2 style kernel on the 16 bit tensor cores, launches are captured into CUDA graphs (see Compiler/README.md for the switches). `GpuMemory.Released` collects dead temporaries and gives unused large pool blocks back between two GPU phases of a program (the VM returns zero).
