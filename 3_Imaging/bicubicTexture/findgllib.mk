################################################################################
#
# Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
#  findgllib.mk is used to find the necessary GL Libraries for specific distributions
#               this is supported on Mac OSX and Linux Platforms
#
################################################################################

# Determine OS platform and unix distribution
ifeq ("$(TARGET_OS)","linux")
   # first search lsb_release
   DISTRO  = $(shell lsb_release -i -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
   DISTVER = $(shell lsb_release -r -s 2>/dev/null)
   ifeq ("$(DISTRO)","")
     # second search and parse /etc/issue
     DISTRO = $(shell more /etc/issue | awk '{print $$1}' | sed '1!d' | sed -e "/^$$/d" 2>/dev/null | tr "[:upper:]" "[:lower:]")
     DISTVER= $(shell more /etc/issue | awk '{print $$2}' | sed '1!d' 2>/dev/null
     # ensure data from /etc/issue is valid
     ifneq (,$(filter-out $(DISTRO),ubuntu fedora red rhel centos suse))
       DISTRO = 
     endif
     ifeq ("$(DISTRO)","")
       # third, we can search in /etc/os-release or /etc/{distro}-release
       DISTRO = $(shell awk '/ID/' /etc/*-release | sed 's/ID=//' | grep -v "VERSION" | grep -v "ID" | grep -v "DISTRIB")
       DISTVER= $(shell awk '/DISTRIB_RELEASE/' /etc/*-release | sed 's/DISTRIB_RELEASE=//' | grep -v "DISTRIB_RELEASE")
     endif
   endif
endif

ifeq ("$(TARGET_OS)","linux")
    # $(info) >> findgllib.mk -> LINUX path <<<)
    # Each set of Linux Distros have different paths for where to find their OpenGL libraries reside
    UBUNTU_PKG_NAME = "nvidia-352"
        UBUNTU = $(shell echo $(DISTRO) | grep -i ubuntu      >/dev/null 2>&1; echo $$?)
        FEDORA = $(shell echo $(DISTRO) | grep -i fedora      >/dev/null 2>&1; echo $$?)
        RHEL   = $(shell echo $(DISTRO) | grep -i 'red\|rhel' >/dev/null 2>&1; echo $$?)
        CENTOS = $(shell echo $(DISTRO) | grep -i centos      >/dev/null 2>&1; echo $$?)
        SUSE   = $(shell echo $(DISTRO) | grep -i suse        >/dev/null 2>&1; echo $$?)
    ifeq ("$(UBUNTU)","0")
      ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        GLPATH := /usr/arm-linux-gnueabihf/lib
        GLLINK := -L/usr/arm-linux-gnueabihf/lib
        ifneq ($(TARGET_FS),) 
          GLPATH += $(TARGET_FS)/usr/lib/$(UBUNTU_PKG_NAME)
          GLPATH += $(TARGET_FS)/usr/lib/arm-linux-gnueabihf
          GLLINK += -L$(TARGET_FS)/usr/lib/$(UBUNTU_PKG_NAME)
          GLLINK += -L$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif 
      else ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-ppc64le)
        GLPATH := /usr/powerpc64le-linux-gnu/lib
        GLLINK := -L/usr/powerpc64le-linux-gnu/lib
      else
        GLPATH    ?= /usr/lib/$(UBUNTU_PKG_NAME)
        GLLINK    ?= -L/usr/lib/$(UBUNTU_PKG_NAME)
        DFLT_PATH ?= /usr/lib
      endif
    endif
    ifeq ("$(SUSE)","0")
      GLPATH    ?= /usr/X11R6/lib64
      GLLINK    ?= -L/usr/X11R6/lib64
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(FEDORA)","0")
      GLPATH    ?= /usr/lib64/nvidia
      GLLINK    ?= -L/usr/lib64/nvidia
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(RHEL)","0")
      GLPATH    ?= /usr/lib64/nvidia
      GLLINK    ?= -L/usr/lib64/nvidia
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(CENTOS)","0")
      GLPATH    ?= /usr/lib64/nvidia
      GLLINK    ?= -L/usr/lib64/nvidia
      DFLT_PATH ?= /usr/lib64
    endif
  
  # find libGL, libGLU, libXi, 
  GLLIB  := $(shell find -L $(GLPATH) $(DFLT_PATH) -name libGL.so  -print 2>/dev/null)
  GLULIB := $(shell find -L $(GLPATH) $(DFLT_PATH) -name libGLU.so -print 2>/dev/null)
  X11LIB := $(shell find -L $(GLPATH) $(DFLT_PATH) -name libX11.so -print 2>/dev/null)

  ifeq ("$(GLLIB)","")
      $(info >>> WARNING - libGL.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif
  ifeq ("$(GLULIB)","")
      $(info >>> WARNING - libGLU.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif
  ifeq ("$(X11LIB)","")
      $(info >>> WARNING - libX11.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif

  HEADER_SEARCH_PATH ?= $(TARGET_FS)/usr/include
  ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
      HEADER_SEARCH_PATH += /usr/arm-linux-gnueabihf/include
  endif

  GLHEADER  := $(shell find -L $(HEADER_SEARCH_PATH) -name gl.h -print 2>/dev/null)
  GLUHEADER := $(shell find -L $(HEADER_SEARCH_PATH) -name glu.h -print 2>/dev/null)
  X11HEADER := $(shell find -L $(HEADER_SEARCH_PATH) -name Xlib.h -print 2>/dev/null)

  ifeq ("$(GLHEADER)","")
      $(info >>> WARNING - gl.h not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif
  ifeq ("$(GLUHEADER)","")
      $(info >>> WARNING - glu.h not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif
  ifeq ("$(X11HEADER)","")
      $(info >>> WARNING - Xlib.h not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif
else
    # This would be the Mac OS X path if we had to do anything special
endif

