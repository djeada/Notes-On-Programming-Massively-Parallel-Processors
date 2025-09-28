# OpenMP Examples Makefile

# C Compiler
CC = gcc

# Compiler flags
CFLAGS = -fopenmp -O3 -std=c99 -Wall

# Debug flags
DEBUG_FLAGS = -g -DDEBUG

# Source files
SOURCES = basic_openmp.c

# Executables
EXECUTABLES = $(SOURCES:.c=)

# Default target
all: $(EXECUTABLES)

# Individual targets
basic_openmp: basic_openmp.c
	$(CC) $(CFLAGS) $< -o $@ -lm

# Debug versions
debug: CFLAGS += $(DEBUG_FLAGS)
debug: $(EXECUTABLES)

# Clean target
clean:
	rm -f $(EXECUTABLES)

# Test target
test: all
	@echo "Testing OpenMP basic example..."
	@./basic_openmp || echo "OpenMP test failed"

# Check OpenMP support
check:
	@echo "Checking OpenMP support..."
	@$(CC) -fopenmp -dM -E - < /dev/null | grep -i openmp || echo "OpenMP not supported"
	@echo "Number of processors: $$(nproc)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all examples"
	@echo "  debug        - Build debug versions"  
	@echo "  clean        - Remove built executables"
	@echo "  test         - Run all examples"
	@echo "  check        - Check OpenMP support"
	@echo "  help         - Show this help message"

.PHONY: all debug clean test check help