EXECUTABLE := harrisCorner serialHarrisCorner
LDFLAGS=-L/opt/cuda/lib64/ -L/usr/local/cuda/lib64 -lcudart
CU_FILES   := harrisCorner.cu
CU_DEPS    :=
CC_FILES   := serialCornerDetection.cpp
LOGS	   := logs
INCL 	   := -I/usr/include/opencv4/ -I/opt/cuda/include -I/usr/local/cuda/include

all: $(EXECUTABLE)

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 -ccbin /usr/bin/gcc
# --gpu-architecture compute_61
LIBS += GL cudart opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

CUDA_OBJS=$(OBJDIR)/harrisCorner.o $(OBJDIR)/harrisCornerMain.o
SERIAL_OBJS=$(OBJDIR)/serialCornerDetection.o


.PHONY: dirs clean

default: harrisCorner

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) 

export: $(EXFILES)
	cp -p $(EXFILES) $(STARTER)

#$(EXECUTABLE): dirs $(OBJS)
#		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

harrisCorner: dirs $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(CUDA_OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

serialHarrisCorner: dirs $(SERIAL_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(SERIAL_OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) $(INCL) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
