// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#spirv_target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformArithmetic, GroupNonUniformBallot],
                     [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
    #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64>
  >
}>

module attributes {transform.with_named_sequence} {
  util.func private @argmax_1d_f32_entry_point(%arg0: tensor<1x32000xf32>) -> tensor<1xi64> {
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<1x32000xf32>
    // Note: This is not safe if the dim size exceeds INT32_MAX. To pass a 64
    // bit value it must be broken down into two 32-bit values for the high and
    // low bits.
    %dim_i32 = arith.index_cast %dim : index to i32
    // Inline external dispatch that conforms to the ABI that the kernel
    // requires. This is the primary reason for the surrounding function as
    // details like tensor shape and push constants need to line up after
    // splicing in the custom dispatch. This allows the kernel author to manage
    // such details by hand without needing the rewrite patterns to worry about
    // things like order of push constants.
    %4 = hal.dispatch.extern "main"[%dim](%dim_i32, %arg0) : (i32, tensor<1x32000xf32>) -> tensor<1xi64>
      count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        hal.return %c1_0, %c1_0, %c1_0 : index, index, index
      }   
      layout(#hal.pipeline.layout<push_constants = 1, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>, 
        #hal.interface.binding<0, 1>
      ])  
      objects({
        #spirv_target ordinal(0) = [ 
          #hal.executable.object<{
            data = dense<"0x03022307000601000b000d0078000000000000001100020001000000110002003d000000110002003f00000011000200400000000b00060001000000474c534c2e7374642e343530000000000e00030000000000010000000f000b0005000000040000006d61696e000000000d00000013000000160000001e0000002a000000710000004b01060004000000260000000700000008000000080000000300030002000000c201000004000900474c5f4558545f636f6e74726f6c5f666c6f775f61747472696275746573000004000a00474c5f474f4f474c455f6370705f7374796c655f6c696e655f646972656374697665000004000800474c5f474f4f474c455f696e636c7564655f6469726563746976650004000a00474c5f4b48525f7368616465725f73756267726f75705f61726974686d6574696300000004000900474c5f4b48525f7368616465725f73756267726f75705f62616c6c6f7400000004000900474c5f4b48525f7368616465725f73756267726f75705f62617369630000000005000400040000006d61696e00000000050004000a0000007468726561645800050008000d000000676c5f4c6f63616c496e766f636174696f6e49440000000005000400120000006c616e65494400000500080013000000676c5f53756267726f7570496e766f636174696f6e49440005000500150000006c616e65436f756e740000000500060016000000676c5f53756267726f757053697a6500050004001a0000006c616e654d617800050005001c000000496e70757442756666657200060005001c000000000000006461746100000000050004001e000000496e70757400000005000500250000006c616e65526573756c74000005000500270000006e756d426174636865730000050006002800000050757368436f6e7374616e7473000000060006002800000000000000746f74616c436f756e740000050003002a00000000000000050003003100000069000000050003003d0000006964780005000400440000006e65775f696e0000050004005300000077674d6178000000050003005800000065710000050004005e00000062616c6c6f74000005000300610000006c7362000500050064000000757070657233326269747300050006006f0000004f757470757442756666657200000000060005006f00000000000000646174610000000005000400710000004f75747075740000470004000d0000000b0000001b00000047000300130000000000000047000400130000000b0000002900000047000300140000000000000047000300160000000000000047000400160000000b00000024000000470003001700000000000000470004001b0000000600000004000000480005001c000000000000002300000000000000470003001c00000002000000470004001e0000002200000000000000470004001e00000021000000000000004800050028000000000000002300000000000000470003002800000002000000480005006f000000000000002300000000000000470003006f0000000200000047000400710000002200000000000000470004007100000021000000010000001300020002000000210003000300000002000000150004000600000020000000000000002b0004000600000007000000400000002b00040006000000080000000100000020000400090000000700000006000000170004000b0000000600000003000000200004000c000000010000000b0000003b0004000c0000000d000000010000002b000400060000000e00000000000000200004000f00000001000000060000003b0004000f00000013000000010000003b0004000f0000001600000001000000160003001800000020000000200004001900000007000000180000001d0003001b000000180000001e0003001c0000001b000000200004001d0000000c0000001c0000003b0004001d0000001e0000000c000000150004001f00000020000000010000002b0004001f000000200000000000000020000400220000000c000000180000001e0003002800000006000000200004002900000009000000280000003b000400290000002a00000009000000200004002b00000009000000060000002000040030000000070000001f0000002b0004001f0000003200000001000000140002003b0000002b0004000600000055000000030000002000040057000000070000003b000000170004005c0000000600000004000000200004005d000000070000005c000000170004006e00000006000000020000001e0003006f0000006e00000020000400700000000c0000006f0000003b00040070000000710000000c00000020000400750000000c0000006e0000002c0006000b000000770000000700000008000000080000003600050002000000040000000000000003000000f8000200050000003b000400090000000a000000070000003b0004000900000012000000070000003b0004000900000015000000070000003b000400190000001a000000070000003b0004000900000025000000070000003b0004000900000027000000070000003b0004003000000031000000070000003b000400090000003d000000070000003b0004001900000044000000070000003b0004001900000053000000070000003b0004005700000058000000070000003b0004005d0000005e000000070000003b0004000900000061000000070000003b000400090000006400000007000000410005000f000000100000000d0000000e0000003d0004000600000011000000100000003e0003000a000000110000003d0004000600000014000000130000003e00030012000000140000003d0004000600000017000000160000003e00030015000000170000003d0004000600000021000000120000004100060022000000230000001e00000020000000210000003d0004001800000024000000230000003e0003001a000000240000003d0004000600000026000000120000003e0003002500000026000000410005002b0000002c0000002a000000200000003d000400060000002d0000002c0000003d000400060000002e0000001500000086000500060000002f0000002d0000002e0000003e000300270000002f0000003e0003003100000032000000f900020033000000f800020033000000f6000400350000003600000000000000f900020037000000f8000200370000003d0004001f00000038000000310000007c0004000600000039000000380000003d000400060000003a00000027000000b00005003b0000003c000000390000003a000000fa0004003c0000003400000035000000f8000200340000003d000400060000003e000000150000003d0004001f0000003f000000310000007c00040006000000400000003f0000008400050006000000410000003e000000400000003d00040006000000420000001200000080000500060000004300000041000000420000003e0003003d000000430000003d00040006000000450000003d0000004100060022000000460000001e00000020000000450000003d0004001800000047000000460000003e00030044000000470000003d0004001800000048000000440000003d00040018000000490000001a000000ba0005003b0000004a00000048000000490000003d000400060000004b0000003d0000003d000400060000004c00000025000000a9000600060000004d0000004a0000004b0000004c0000003e000300250000004d0000003d000400180000004e0000001a0000003d000400180000004f000000440000000c000700180000005000000001000000280000004e0000004f0000003e0003001a00000050000000f900020036000000f8000200360000003d0004001f0000005100000031000000800005001f0000005200000051000000320000003e0003003100000052000000f900020033000000f8000200350000003d00040018000000540000001a0000006601060018000000560000005500000000000000540000003e00030053000000560000003d0004001800000059000000530000003d000400180000005a0000001a000000b40005003b0000005b000000590000005a0000003e000300580000005b0000003d0004003b0000005f00000058000000530105005c00000060000000550000005f0000003e0003005e000000600000003d0004005c000000620000005e00000057010500060000006300000055000000620000003e00030061000000630000003e000300640000000e0000003d0004000600000065000000120000003d000400060000006600000061000000aa0005003b0000006700000065000000660000003d00040006000000680000000a0000003d000400060000006900000015000000b00005003b0000006a0000006800000069000000a70005003b0000006b000000670000006a000000f70003006d00000000000000fa0004006b0000006c0000006d000000f80002006c0000003d0004000600000072000000250000003d000400060000007300000064000000500005006e00000074000000720000007300000041000500750000007600000071000000200000003e0003007600000074000000f90002006d000000f80002006d000000fd00010038000100"> : vector<3392xi8>
          }>
        ]
      })
    util.return %4 : tensor<1xi64>
  }

  // Custom matcher for argmax operations equivalent to the custom kernel. This
  // matcher will be run one-by-one on all operations contained within the
  // target function. On success, it will return the handle to the matched
  // argmax operation.
  transform.named_sequence @match_argmax(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    // Fail fast on non-linalg generics.
    transform.match.operation_name %generic ["linalg.generic"] : !transform.any_op
    %matched = transform.match.structured failures(propagate) %generic : (!transform.any_op) -> (!transform.any_op) {
    ^bb1(%argmax: !transform.any_op):
      // Verify that the rank (i.e. number of loops) of the linalg op is 2,
      // with one parallel iterator and one reduction iterator.
      // TODO: Add optionality for the parallel dimensions.
      %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %argmax : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c2 : !transform.param<i64>
      transform.match.structured.dim %argmax[0] {parallel} : !transform.any_op
      transform.match.structured.dim %argmax[-1] {reduction} : !transform.any_op

      // Verify a single input (target vector to compute the argmax of) and two
      // outputs, one for the maximum value and one for the index.
      %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
      %n_inputs = transform.match.structured.num_inputs %argmax : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %n_inputs, %c1 : !transform.param<i64>
      %n_outputs = transform.match.structured.num_inits %argmax : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %n_outputs, %c2 : !transform.param<i64>
  
      transform.match.structured.yield %argmax : !transform.any_op 
    }

    // Verify the operand shapes of the linalg op. For example, in the below,
    // dim 0 must be statically 1, and dim 1 must be statically divisible by 64.
    %in0 = transform.get_operand %matched[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<1x?xf32> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[1], 64 : !transform.any_value
    %out0 = transform.get_operand %matched[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %out0 = tensor<1xf32> : !transform.any_value
    %out1 = transform.get_operand %matched[2] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %out1 = tensor<1xi64> : !transform.any_value

    // Verify the region of the argmax op. This does a structural comparison of
    // region(s) of the payload operation against the single operation contained
    // within the body of this operation. This does no verification of other
    // input types/attributes. This is because typically for kernel matching,
    // the most important part to get exactly right is the inner loop. Otherwise
    // small variations to shape information and iterator counts and such are
    // better suited for more general matchers.
    transform.iree.match.regions %matched : !transform.any_op {
      ^bb0(%target: tensor<1x?xf32>, %empty_max: tensor<1xf32>, %empty_idx: tensor<1xi64>):
        %5:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                                affine_map<(d0, d1) -> (d0)>,
                                                affine_map<(d0, d1) -> (d0)>],
                               iterator_types = ["parallel", "reduction"]}
                               ins(%target : tensor<1x?xf32>)
                               outs(%empty_max, %empty_idx : tensor<1xf32>, tensor<1xi64>) {
        ^bb0(%in: f32, %out: f32, %out_0: i64):
          %6 = linalg.index 1 : index
          %7 = arith.index_cast %6 : index to i64
          %8 = arith.maximumf %in, %out : f32
          %9 = arith.cmpf ogt, %in, %out : f32
          %10 = arith.select %9, %7, %out_0 : i64
          linalg.yield %8, %10 : f32, i64
        } -> (tensor<1xf32>, tensor<1xi64>)
    }
    transform.yield %generic : !transform.any_op
  }

  // Rewrite callback for `transform.foreach_match`. The input signature for
  // this sequence must match exactly with the outputs of the matcher. In this
  // case we just take the argmax as an input, import the entry point for the
  // custom kernel authored above, and replace the users of the argmax with a
  // call to the function.
  transform.named_sequence @cast_and_call_argmax(%argmax: !transform.any_op {transform.readonly}) {
    %module = transform.util.get_nearest_symbol_table %argmax : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @argmax_1d_f32_entry_point into %module if undefined : (!transform.any_op) -> !transform.any_op
    %ins = transform.get_operand %argmax[0] : (!transform.any_op) -> !transform.any_value
    %outs = transform.get_result %argmax[1] : (!transform.any_op) -> !transform.any_value
    transform.util.cast_and_call %func(%ins) -> %outs before %argmax {
          // This specifies how to resolve type mismatches between the arguments
          // of the function and the inputs to the argmax. In this example, the
          // only casts this will generate are same-rank tensor casts that drop
          // static information.
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  // Entry point for the transform interpreter, nested on the full module. This
  // is because the rewrites needed for importing the custom kernel needs to
  // add a new symbol to the module's symbol table.
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    // Gather the set of functions within the module.
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op   
    // For each function in the module, run the matcher on all contained
    // operations.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            // <matcher name> -> <rewriter name>
            // Multiple matcher-action pairs can be specified comma separated,
            // here we are only doing a single kind of match and replace.
            //
            // Note that the operations within the module are walked in
            // post-order, meaning actions must be very careful in their
            // replacements not to modify successors of operations. Nested
            // regions and DAG roots will be visited last so it is safest to
            // do matching + replacement on the root of the DAG rather than
            // trying to look ahead. The other option is to avoid dce/cse until
            // after the walk is complete.
            @match_argmax -> @cast_and_call_argmax
          : (!transform.any_op) -> (!transform.any_op)
    }
    // Cleanup now dead instances of argmax.
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
