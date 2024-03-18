# macroHipKernel

## Instructions to compile and test argmax:

```sh
./generate_hsaco.sh argmax_ukernel.c
hipcc argmax.cpp
./a.out 32000
```