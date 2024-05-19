# Kernel Kitchen

Prepare and cook compute kernels without heating up the GPU

> [!IMPORTANT]
>
> This fun little project is meant for educational purposes first and
> foremost. Experiment around, add more kernels, or implement missing
> features. Just don't expect too much. And importantly, ignore performance
> unless you do heat up the GPU after all.

## Example

```console
$ make examples/hello
(...)
$ examples/hello
Found 1 device
Device 0 name:	VGPU
Launching kernel hello_1D<<<dim3(1, 1, 1), dim3(8, 1, 1)>>>
Hello from Thread 0
Hello from Thread 3
Hello from Thread 2
Hello from Thread 4
Hello from Thread 7
Hello from Thread 1
Hello from Thread 6
Hello from Thread 5
Launching kernel hello_2D<<<dim3(1, 1, 1), dim3(4, 2, 1)>>>
Hello from Thread (0, 0)
Hello from Thread (2, 1)
Hello from Thread (3, 1)
Hello from Thread (2, 0)
Hello from Thread (3, 0)
Hello from Thread (1, 1)
Hello from Thread (0, 1)
Hello from Thread (1, 0)
Launching kernel hello_3D<<<dim3(1, 1, 1), dim3(4, 2, 1)>>>
Hello from Thread (0, 0, 0)
Hello from Thread (1, 1, 0)
Hello from Thread (3, 0, 0)
Hello from Thread (3, 1, 0)
Hello from Thread (1, 0, 0)
Hello from Thread (2, 0, 0)
Hello from Thread (2, 1, 0)
Hello from Thread (0, 1, 0)
```

Note that the order of output may differ between runs.

```console
$ make help

Usage:
  make (all)           Build all examples
  make check           Run FileChecks
  make compdb          Generate a compilation database (default: compile_commands.json)
  make CUDAFY=1        Convert all examples to CUDA
  make CUDA=1          Build CUDA versions of all examples
  make HIPIFY=1        Convert all examples to HIP
  make HIP=1           Build HIP versions of all examples
```

## References

- J. Sanders and E. Kandrot. *CUDA by Example: An Introduction to General-Purpose GPU Programming*.
- W. M. W. Hwu, D. B. Kirk, and I. E. Hajj. *Programming Massively Parallel Processors: A Hands-on Approach*.
