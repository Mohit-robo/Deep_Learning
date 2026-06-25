# Makefile for compiling a DeepStream Dfine Bbox Parser

# Default location of DeepStream SDK
DEEPSTREAM_PATH ?= /opt/nvidia/deepstream/deepstream

# Default location of CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# Compiler
CC := g++

# Target shared library
TARGET_LIB := libdfineparser.so

# Source files
SRCS := dfine_parse.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)

# Include directories
INCLUDES = -I$(DEEPSTREAM_PATH)/sources/includes \
           -I$(CUDA_PATH)/include

# C++ compiler flags
# -O2: Optimization level
# -fPIC: Generate position-independent code, required for shared libraries
# -Wall: Enable all warnings
# -std=c++11: Use the C++11 standard
CXXFLAGS = -O2 -fPIC -Wall -std=c++11 $(INCLUDES)

# Linker flags
# -shared: Create a shared library
# -Wl,-soname,$@: Set the "soname" of the library
LDFLAGS = -shared -Wl,-soname,$(TARGET_LIB)

# Libraries to link against (if any were needed, but for this simple parser, none are)
LIBS =

# Default target
all: $(TARGET_LIB)

# Rule to link the shared library
$(TARGET_LIB): $(OBJS)
	@echo "--- Linking shared library: $(TARGET_LIB) ---"
	$(CC) $(LDFLAGS) -o $(TARGET_LIB) $(OBJS) $(LIBS)
	@echo "Successfully created $(TARGET_LIB)"

# Rule to compile C++ source files into object files
%.o: %.cpp
	@echo "--- Compiling: $< ---"
	$(CC) $(CXXFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	@echo "--- Cleaning up ---"
	rm -f $(OBJS) $(TARGET_LIB)
	@echo "Clean complete"

# Phony targets
.PHONY: all clean