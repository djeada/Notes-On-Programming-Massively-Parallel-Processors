# CUDA Examples Makefile

# CUDA Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++11

# Debug flags
DEBUG_FLAGS = -g -G -DDEBUG

# Source files
SOURCES = vector_addition.cu matrix_multiplication.cu reduction.cu

# Executables
EXECUTABLES = $(SOURCES:.cu=)

# Default target
all: $(EXECUTABLES)

# Individual targets
vector_addition: vector_addition.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

matrix_multiplication: matrix_multiplication.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

reduction: reduction.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Debug versions
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: $(EXECUTABLES)

# Clean target
clean:
	rm -f $(EXECUTABLES)

# Test target (requires CUDA-capable GPU)
test: all
	@echo "Testing vector addition..."
	@./vector_addition || echo "Vector addition test failed (GPU required)"
	@echo "Testing matrix multiplication..."
	@./matrix_multiplication || echo "Matrix multiplication test failed (GPU required)"
	@echo "Testing reduction..."
	@./reduction || echo "Reduction test failed (GPU required)"

# Print GPU information
info:
	@echo "CUDA Device Information:"
	@nvidia-smi || echo "nvidia-smi not available"
	@nvcc --version || echo "NVCC not available"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all examples"
	@echo "  debug        - Build debug versions"
	@echo "  clean        - Remove built executables"
	@echo "  test         - Run all examples (requires GPU)"
	@echo "  info         - Display GPU and CUDA information"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Individual examples:"
	@echo "  vector_addition"
	@echo "  matrix_multiplication"
	@echo "  reduction"

.PHONY: all debug clean test info help