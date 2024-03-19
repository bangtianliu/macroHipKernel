~/nod/iree-build-notrace/llvm-project/bin/clang -x hip --offload-arch=gfx1100 --offload-device-only -nogpulib -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -O3 -fvisibility=protected -emit-llvm -c $1 -o $1.bc
~/nod/iree-build-notrace/llvm-project/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1100 -c $1.bc -o $1.o
~/nod/iree-build-notrace/llvm-project/bin/clang -target amdgcn-amd-amdhsa $1.o -o $1.hsaco
xxd -p -c 1000000 $1.hsaco > $1.hex
rm $1.bc $1.o
