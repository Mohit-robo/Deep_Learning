CUDA_VER ?= 12.6
DEEPSTREAM_DIR ?= /opt/nvidia/deepstream/deepstream-7.1

CXX ?= g++

TARGET = libpostprocess.so
SOURCES = postprocess.cpp

DEEPSTREAM_INCLUDE = $(DEEPSTREAM_DIR)/sources/includes

CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64


CPPFLAGS += \
    -I$(DEEPSTREAM_INCLUDE) \
    -I$(CUDA_INCLUDE)

CXXFLAGS += \
    -std=c++17 \
    -O3 \
    -fPIC \
    -Wall \
    -Wextra \
    -Wno-unused-parameter

LDFLAGS += \
    -shared \
    -L$(CUDA_LIB)

LIBS += \
    -lcudart

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	-o $@ $(SOURCES) \
	$(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
