~/nod/iree-build-notrace/install/bin/iree-compile ~/nod/macroHipKernel/example_transform.mlir --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=rocm --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --iree-input-type=torch --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --iree-llvmcpu-target-cpu-features=host --iree-llvmcpu-target-triple=x86_64-linux-gnu --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 --iree-vulkan-target-triple=rdna3-unknown-linux --iree-preprocessing-transform-spec-filename=example_transform_spec.mlir --iree-stream-resource-max-allocation-size=4294967296 -o ~/nod/macroHipKernel/argmax.vmfb

~/nod/iree-build-notrace/install/bin/iree-benchmark-module --module=argmax.vmfb --function=mixed_invocation --input=1x128xf32=2.0 --input=1x128xf32=1.0 --device=rocm

rm argmax.vmfb
