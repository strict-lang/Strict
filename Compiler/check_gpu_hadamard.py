"""Check strict_device_hadamard_tokens against numpy; run from Strict with LLVM/MLIR installed and an NVIDIA GPU."""
import ctypes, subprocess, tempfile
import numpy as np
from pathlib import Path
temporary=tempfile.TemporaryDirectory(prefix='strict-hadamard-')
root=Path(temporary.name)
parts=[Path('Compiler/NativeRuntime'+name+'.mlir').read_text(encoding='utf-8') for name in ['Device','Nvidia','Hadamard']]
module='module attributes {llvm.target_triple = "nvptx64-nvidia-cuda"} {\n'+'\n'.join(parts)+'\n}\n'
(root/'hadamard.mlir').write_text(module,encoding='utf-8')
subprocess.run(['C:/msys64/ucrt64/bin/mlir-opt.exe',str(root/'hadamard.mlir'),'--symbol-dce','--convert-scf-to-cf','--convert-arith-to-llvm','--convert-func-to-llvm','--convert-cf-to-llvm','--reconcile-unrealized-casts','-o',str(root/'hadamard.llvm.mlir')],check=True)
subprocess.run(['C:/msys64/ucrt64/bin/mlir-translate.exe','--mlir-disable-threading','--mlir-to-llvmir',str(root/'hadamard.llvm.mlir'),'-o',str(root/'hadamard.ll')],check=True)
subprocess.run(['C:/Program Files/LLVM/bin/clang.exe','--target=nvptx64-nvidia-cuda','-O3','-fno-slp-vectorize','-S','-Wno-override-module','-Xclang','-target-feature','-Xclang','+ptx78','-march=sm_80',str(root/'hadamard.ll'),'-o',str(root/'hadamard.ptx')],check=True)
cuda=ctypes.WinDLL('nvcuda')
def checked(result):
	assert result==0,'CUDA error '+str(result)
context=ctypes.c_void_p();device=ctypes.c_int()
checked(cuda.cuInit(0));checked(cuda.cuDeviceGet(ctypes.byref(device),0))
checked(cuda.cuCtxCreate_v2(ctypes.byref(context),0,device))
ptx=(root/'hadamard.ptx').read_bytes()+b'\0'
loaded=ctypes.c_void_p();kernel=ctypes.c_void_p()
checked(cuda.cuModuleLoadData(ctypes.byref(loaded),ptx))
checked(cuda.cuModuleGetFunction(ctypes.byref(kernel),loaded,b'strict_device_hadamard_tokens'))
def rotated(values):
	for step in [1,4,16,64]:
		groups=values.reshape(-1,256)
		index=np.arange(256)
		digit=index//step%4
		first=index-digit*step
		signs=lambda part:np.where(3-digit==part,np.float32(-0.5),np.float32(0.5))
		quad=[groups[:,first+part*step]*signs(part) for part in range(4)]
		values=(((quad[0]+quad[1])+quad[2])+quad[3]).reshape(-1)
	return values
def expected(numbers,width):
	tokens=rotated(numbers).reshape(-1,width)
	largest=np.abs(tokens).max(axis=1)
	factor=np.float32(127)/largest
	codes=np.rint(tokens*factor[:,None]).astype(np.int8).reshape(-1)
	return codes,np.repeat(largest/np.float32(127),width//32)
def quantized(numbers,width,threads):
	count=numbers.size
	source=ctypes.c_uint64();target=ctypes.c_uint64()
	checked(cuda.cuMemAlloc_v2(ctypes.byref(source),ctypes.c_size_t(count*4)))
	checked(cuda.cuMemAlloc_v2(ctypes.byref(target),ctypes.c_size_t(count+count//8)))
	checked(cuda.cuMemsetD8_v2(target,0,ctypes.c_size_t(count+count//8)))
	checked(cuda.cuMemcpyHtoD_v2(source,ctypes.c_void_p(numbers.ctypes.data),ctypes.c_size_t(count*4)))
	values=[source,ctypes.c_uint64(count),ctypes.c_uint64(width),target,ctypes.c_uint64(32)]
	arguments=(ctypes.c_void_p*5)(*[ctypes.addressof(value) for value in values])
	checked(cuda.cuLaunchKernel(kernel,count//width,1,1,threads,1,1,0,None,arguments,None))
	checked(cuda.cuCtxSynchronize())
	result=np.zeros(count+count//8,np.uint8)
	checked(cuda.cuMemcpyDtoH_v2(ctypes.c_void_p(result.ctypes.data),target,ctypes.c_size_t(result.size)))
	checked(cuda.cuMemFree_v2(source));checked(cuda.cuMemFree_v2(target))
	return result[:count].view(np.int8),result[count:].view(np.float32)
generator=np.random.default_rng(7)
for width,threads in [(5376,672),(14336,896)]:
	numbers=generator.standard_normal(3*width).astype(np.float32)
	codes,scales=quantized(numbers,width,threads)
	wantedCodes,wantedScales=expected(numbers,width)
	assert np.array_equal(codes,wantedCodes),('codes of tokens of '+str(width),int((codes!=wantedCodes).sum()))
	assert np.array_equal(scales,wantedScales),('scales of tokens of '+str(width),int((scales!=wantedScales).sum()))
	print('strict_device_hadamard_tokens of 3 tokens of',width,'with',threads,'threads matches numpy',flush=True)
checked(cuda.cuCtxDestroy_v2(context))
temporary.cleanup()
