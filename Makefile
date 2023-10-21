CC := clang
CFLAGS += -Wall -Wextra -fopenmp -O2
CPPFLAGS += -DDEBUG -Iinclude
LDFLAGS += -fopenmp

SRCS := $(wildcard examples/*.c)
OBJS := $(SRCS:.c=.o)
EXES := $(OBJS:.o=)

have = $(if $(shell command -v $(1) 2> /dev/null),1)

## make (all): Build all examples
all: $(EXES)

$(OBJS): include/vgpu.h include/common.h Makefile

## make check: Run FileChecks
ifneq ($(call have,FileCheck),)
check: $(EXES)
	@for exe in $(sort $^); do \
	    echo "./run.sh $$exe | FileCheck $$exe.c"; \
	    ./run.sh $$exe | FileCheck $$exe.c; \
	done
else
check:
	@echo "Requires FileCheck: https://llvm.org/docs/CommandGuide/FileCheck.html"
endif

%: %.cu
	$(NVCC) -Iexamples $< -o $@

examples/cuda/%.cu: examples/%.c | examples/cuda
	lua cudafy.lua $< > $@

%: %.cpp
	$(HIPCC) -Iexamples $< -o $@

examples/hip/%.cpp: examples/cuda/%.cu | examples/hip
	hipify-perl $< > $@
	@diff -u $< $@ > $@.diff; [ $$? = 1 ]

examples/cuda examples/hip:
	@mkdir -p $@
	@touch $@/vgpu.h
	@printf "#ifndef VGPU_H\n" >> $@/vgpu.h
	@printf "#define VGPU_H\n" >> $@/vgpu.h
	@printf "\n#include \"common.h\"\n\n" >> $@/vgpu.h
	@printf "#endif // VGPU_H\n" >> $@/vgpu.h
	@cp include/common.h $@

clean::
	$(RM) $(OBJS) $(EXES)

realclean: clean
	$(RM) -r examples/cuda examples/hip

COMPDB ?= compile_commands.json
## make compdb: Generate a compilation database (default: compile_commands.json)
ifneq ($(call have,bear),)
compdb: $(COMPDB)
$(COMPDB): Makefile
	@$(MAKE) --no-print-directory clean
	bear -o $(COMPDB) $(MAKE) --no-print-directory
clean::
	$(RM) $(COMPDB)
else
compdb:
	@echo "Requires Bear: https://github.com/rizsotto/Bear"
endif

## make CUDAFY=1: Convert all examples to CUDA
ifeq ($(CUDAFY), 1)
CUDA_SRCS := $(patsubst %.c,%.cu,$(subst /,/cuda/,$(SRCS)))
all: $(CUDA_SRCS)
clean::
	$(RM) $(CUDA_SRCS)
endif

## make CUDA=1: Build CUDA versions of all examples
ifeq ($(CUDA), 1)
NVCC := nvcc
CUDA_EXES := $(subst /,/cuda/,$(EXES))
all: $(CUDA_EXES)
clean::
	$(RM) $(CUDA_EXES)
endif

## make HIPIFY=1: Convert all examples to HIP
ifeq ($(HIPIFY), 1)
HIP_SRCS := $(patsubst %.c,%.cpp,$(subst /,/hip/,$(SRCS)))
all: $(HIP_SRCS)
clean::
	$(RM) $(HIP_SRCS)
endif

## make HIP=1: Build HIP versions of all examples
ifeq ($(HIP), 1)
HIP_PLATFORM ?= nvidia
HIP_PATH ?= $(HOME)/HIP/nvidia
HIPCC := $(HIP_PATH)/bin/hipcc
ifeq ($(HIP_PLATFORM), nvidia)
  CUDA_PATH ?= /opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda
  HIP_ENV := HIP_PLATFORM=$(HIP_PLATFORM)
  HIP_ENV += CUDA_PATH=$(CUDA_PATH)
  HIPCC := $(HIP_ENV) $(HIPCC)
endif
HIP_EXES := $(subst /,/hip/,$(EXES))
all: $(HIP_EXES)
clean::
	$(RM) $(HIP_EXES)
endif

help: Makefile
	@echo
	@echo "Usage:"
	@awk '\
	BEGIN { FS = ":" } \
	/^##/ { \
	    i = index($$0, ":"); \
	    match($$1, /^##\s*/); \
	    printf("  %-20s%s\n", substr($$1, RSTART + RLENGTH), substr($$0, i + 1)); \
	}' $<
	@echo

.PHONY: all check clean compdb help realclean
.PRECIOUS: examples/cuda/%.cu examples/hip/%.cpp
