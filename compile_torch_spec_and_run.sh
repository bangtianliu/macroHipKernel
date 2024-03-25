python value_generator.py

/home/stanley/nod/iree-build-notrace/compiler/bindings/python/iree/compiler/tools/../_mlir_libs/iree-compile torch_example_transform.mlir --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=rocm --mlir-print-op-on-diagnostic=false --iree-input-type=torch --mlir-print-op-on-diagnostic=false --iree-llvmcpu-target-cpu-features=host --iree-llvmcpu-target-triple=x86_64-linux-gnu --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 --iree-rocm-target-chip=gfx1100 --iree-vm-bytecode-module-strip-source-map=true --iree-opt-strip-assertions=true --iree-vm-target-truncate-unsupported-floats --iree-codegen-llvmgpu-enable-transform-dialect-jit=false --iree-preprocessing-transform-spec-filename=/home/stanley/nod/macroHipKernel/example_transform_spec.mlir -o torch_argmax.vmfb

~/nod/iree-build-notrace/install/bin/iree-run-module --module=torch_argmax.vmfb --function=argmax --input=@argmax_3d_input_f16.npy --device=rocm --expected_output=@argmax_3d_output_f16.npy
~/nod/iree-build-notrace/install/bin/iree-benchmark-module --module=torch_argmax.vmfb --function=argmax --input=@argmax_3d_input_f16.npy --device=rocm

rm torch_argmax.vmfb
