EXECUTABLE := harrisCorner serialHarrisCorner
LDFLAGS=-L/opt/cuda/lib64/ -lcudart
CU_FILES   := harrisCorner.cu
CU_DEPS    :=
CC_FILES   := serialCornerDetection.cpp
LOGS	   := logs
INCL 	   := -I/usr/include/opencv4/ -I/opt/cuda/include

all: $(EXECUTABLE)

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
LIBS += GL cudart opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/serialCornerDetection.o $(OBJDIR)/harrisCorner.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) *.ppm

check:	default
		./checker.pl

export: $(EXFILES)
	cp -p $(EXFILES) $(STARTER)


$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)




$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) $(INCL) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@