import numpy as np

batch = 1
reductionSize = 32000 # tried 32000, 32, 10
label = 98
data = np.ones([batch, reductionSize]).astype(np.float32)*7
# data = np.random.normal(size=[batch, reductionSize]).astype(np.float32)
data[0, label] = 53.0
np.save("input0.npy", data)
data_fp16 = data.astype(np.float16)
np.save("input0_f16.npy", data_fp16)

batch0 = 1
batch1 = 1
reductionSize = 32000 # tried 32000, 32, 10
argmax_input = np.random.normal(size=[batch0, batch1, reductionSize]).astype(np.float32)
argmax_output = np.argmax(argmax_input,axis=-1).astype(np.float32)
np.save("argmax_3d_input_f32.npy", argmax_input)
np.save("argmax_3d_output_f32.npy", argmax_output)


argmax_input = np.random.normal(size=[batch0, batch1, reductionSize]).astype(np.float16)
argmax_output = np.argmax(argmax_input,axis=-1).astype(np.float32)
np.save("argmax_3d_input_f16.npy", argmax_input)
np.save("argmax_3d_output_f16.npy", argmax_output)
