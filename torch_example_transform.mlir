// RUN: iree-compile %s \
// RUN:     --iree-hal-executable-object-search-path=$IREE_BINARY_DIR \
// RUN:     --iree-preprocessing-transform-spec-filename=%p/example_transform_spec.mlir | \
// RUN: iree-run-module \
// RUN:     --device=vulkan \
// RUN:     --module=- \
// RUN:     --function=mixed_invocation \
// RUN:     --input=1x128xf32=4 \
// RUN:     --input=1x128xf32=3 | \
// RUN: FileCheck %s

// The configuration used for executable compilation.
// This lets the compiler and runtime know the format and requirements of the
// executable binaries produced and multiple variants with differing formats
// and compilation options (architectures, etc) can be embedded for runtime
// selection.
// HACK: Currently this must match EXACTLY with the executable target for the
// custom kernel. For things to be truly portable, we need to be able to compare
// executable configurations.

// The target devices that the program will run on. We can compile and run with
// multiple targets, but this example is maintaining an implicit requirement
// that the custom kernel being spliced in is supported by the target device,
// hence we only support vulkan here. It is possible to hand author a custom
// kernel that supports multiple targets by specifying an object per-target, but
// that requires authoring the kernel for multiple targets.
  // CHECK-LABEL: EXEC @mixed_invocation
  func.func @argmax(%arg0: tensor<1x?x32000xf16>) -> tensor<1x1xi64> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int6 = torch.constant.int 6
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %64 = torch_c.from_builtin_tensor %arg0 : tensor<1x?x32000xf16> -> !torch.vtensor<[1,?,32000],f16>
    %5771 = torch.prims.convert_element_type %64, %int6 : !torch.vtensor<[1,?,32000],f16>, !torch.int -> !torch.vtensor<[1,?,32000],f32>
    %5772 = torch.aten.select.int %5771, %int1, %int-1 : !torch.vtensor<[1,?,32000],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,32000],f32>
    %5773 = torch.aten.argmax %5772, %int1, %false : !torch.vtensor<[1,32000],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1],si64>
    %5774 = torch.aten.unsqueeze %5773, %int0 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %5775 = torch_c.to_builtin_tensor %5774 : !torch.vtensor<[1,1],si64> -> tensor<1x1xi64>
    return %5775 : tensor<1x1xi64>
  }
