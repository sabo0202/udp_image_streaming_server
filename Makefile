include CommonDefs.mak

BIN_DIR = Bin

SRC_FILES = kinect_eyeDrivev1.cpp PracticalSocket.cpp
#SRC_FILES = main_edited.cpp

ifeq ("$(OSTYPE)","Darwin")
    CFLAGS += -DMACOS
    LDFLAGS += -framework OpenGL -framework GLUT
else
    CFLAGS += -DUNIX -DGLX_GLXEXT_LEGACY
    USED_LIBS += glut GL
endif

USED_LIBS += stdc++ python2.7 OpenNI2 opencv_core opencv_calib3d opencv_features2d opencv_flann opencv_ml opencv_objdetect opencv_highgui opencv_imgproc opencv_imgcodecs opencv_tracking opencv_videoio opencv_rgbd opencv_viz


EXE_NAME = kinect_eyeDrive1
CFLAGS += -Wall

INC_DIRS = \
           ../../../opencv2 \
	   ../../../openni2 \
	   ../../../Include \

include CommonCppMakefile


