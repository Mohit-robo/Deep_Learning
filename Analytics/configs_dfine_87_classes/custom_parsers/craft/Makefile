CXX := g++

SRCS := nvdsinfer_custombboxparser_craft.cpp
INCS := $(wildcard *.h)

LIB := libnvdsinfer_custom_craft.so

CUDA_INC := /usr/local/cuda/include
CUDA_LIB := /usr/local/cuda/lib64

DS_INC := /opt/nvidia/deepstream/deepstream/sources/includes

OPENCV_INC := /usr/local/include/opencv4
OPENCV_LIB := /usr/local/lib

CFLAGS := -O3 -Wall -fPIC -std=c++17 \
           -I$(DS_INC) \
           -I$(CUDA_INC) \
           -I$(OPENCV_INC)

LIBS := -L$(CUDA_LIB) \
        -L$(OPENCV_LIB) \
        -lopencv_core \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -lopencv_cudaimgproc \
        -lopencv_cudaarithm \
        -lopencv_cudafilters \
        -lopencv_cudawarping \
        -lcudart

all: $(LIB)

$(LIB): $(SRCS) $(INCS)
	$(CXX) $(SRCS) $(CFLAGS) $(LIBS) \
	-shared -Wl,-no-undefined \
	-o $@

clean:
	rm -f $(LIB)

install:
	sudo cp $(LIB) /opt/nvidia/deepstream/deepstream/lib/