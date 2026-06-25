CUDA_VER ?= 12.6
DEEPSTREAM_DIR ?= /opt/nvidia/deepstream/deepstream-7.1

CXX ?= g++

TARGET = libnvdsinfer_custom_parseq.so
SOURCES = parseq_sequence_parser.cpp

DEEPSTREAM_INCLUDE = $(DEEPSTREAM_DIR)/sources/includes

CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

OPENCV_INCLUDE = /usr/local/include/opencv4
OPENCV_LIB = /usr/local/lib

CPPFLAGS += \
    -I$(DEEPSTREAM_INCLUDE) \
    -I$(CUDA_INCLUDE) \
    -I$(OPENCV_INCLUDE)

CXXFLAGS += \
    -std=c++17 \
    -O2 \
    -fPIC \
    -Wall \
    -Wextra \
    -Wno-unused-parameter

LDFLAGS += \
    -shared \
    -L$(CUDA_LIB) \
    -L$(OPENCV_LIB)

LIBS += \
    -lcudart \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lopencv_cudaimgproc \
    -lopencv_cudaarithm \
    -lopencv_cudafilters \
    -lopencv_cudawarping

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
	-o $@ $(SOURCES) \
	$(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean