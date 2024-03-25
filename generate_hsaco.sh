rm $1.hsaco $1.hex
~/nod/iree-build-notrace/llvm-project/bin/clang -x hip --offload-arch=gfx1100 --offload-device-only -nogpulib -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -O3 -fvisibility=protected -emit-llvm -c $1 -o $1.bc
~/nod/iree-build-notrace/llvm-project/bin/llvm-link ~/nod/iree-build-notrace/install/lib/iree_platform_libs/rocm/ockl.bc ~/nod/iree-build-notrace/install/lib/iree_platform_libs/rocm/ocml.bc /opt/rocm/amdgcn/bitcode/opencl.bc /opt/rocm/amdgcn/bitcode/hip.bc /opt/rocm/amdgcn/bitcode/oclc_isa_version_1100.bc $1.bc -o $1.linked.bc
~/nod/iree-build-notrace/llvm-project/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1100 -c $1.linked.bc -o $1.o
/opt/rocm/llvm/bin/llvm-objdump --disassemble --arch=amdgcn $1.o > rocm.asm
~/nod/iree-build-notrace/llvm-project/bin/lld -flavor gnu -shared $1.o -o $1.hsaco
xxd -p -c 1000000 $1.hsaco > $1.hex
rm $1.bc $1.o $1.linked.bc
