################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CUDA_VER?=12.6

APP:= deepstream-test5-app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=7.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= deepstream_test5_app_main.c deepstream_utc.c
SRCS+= ../deepstream-app/deepstream_app.c ../deepstream-app/deepstream_app_config_parser.c
SRCS+= ../deepstream-app/deepstream_app_config_parser_yaml.cpp
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/apps/apps-common/src/*.c)
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/apps/apps-common/src/deepstream-yaml/*.cpp)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)

CFLAGS+= -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/apps/apps-common/includes \
		 -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes \
		 -I../deepstream-app/ -DDS_VERSION_MINOR=0 -DDS_VERSION_MAJOR=5 \
		 -I/usr/local/cuda/include

LIBS:= -L/usr/local/cuda/lib64/ -lcudart

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_batch_jpegenc -lnvdsgst_customhelper -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm \
       -lyaml-cpp -lcuda -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(APP)

# deepstream_test5_app_main.c includes a C++ header (plate_accumulators.hpp)
# so it must be compiled with g++ instead of gcc.
deepstream_test5_app_main.o: deepstream_test5_app_main.c $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

# deepstream_utc.c is linked against the C++ main object, so compile
# it with g++ too to avoid C/C++ name mangling mismatches.
deepstream_utc.o: deepstream_utc.c $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)

