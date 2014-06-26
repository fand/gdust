# Makefile
# doc1: http://www.gnu.org/software/make/manual/make.html
# doc2: http://netbsd.gw.com/cgi-bin/man-cgi?make+1+NetBSD-current

CXX      = nvcc
CPX      = g++
CFLAGS   = -Iinclude -use_fast_math -Xcompiler -fopenmp -g -G -arch sm_35
CPPFLAGS = -Iinclude -use_fast_math -O3 -msse2 -msse3 -fopenmp -g -G
# -L../opt/boost/lib -I../opt/boost/include

LIBS     = -lcurand -lgsl -lgslcblas

APPNAME  = bin/gdustdtw

#####################################################################


CUDASRC	= $(wildcard src/*.cu)
CPPSRC	= $(wildcard src/*.cpp)
#HDR	= $(wildcard include/*.hpp)
OBJ	= $(addsuffix .o, $(basename $(CUDASRC))) $(addsuffix .o, $(basename $(CPPSRC)))


all:  header $(APPNAME) trailer


src/%.o: src/%.cu
	@echo Compiling: "$@ ( $< )"
	@$(CXX) $(CFLAGS) -c -o $@ $<

src/%.o: src/%.cpp
	@echo Compiling: "$@ ( $< )"
	@$(CPX) $(CPPFLAGS) -c -o $@ $<

$(APPNAME): $(OBJ)
	@echo Compiling: "$@ ( $^ )"
	@$(CXX) $(CFLAGS) $(OBJ) -o $(APPNAME) $(LIBS)


header:
	@echo "%"
	@echo "%  Compiling $(APPNAME)"
	@echo "%"
	@echo "%  CXX..................: $(CXX)"
	@echo "%  CFLAGS...............: $(CFLAGS)"
	@echo "%  LIBS.................: $(LIBS)"
	@echo "%"

trailer:
	@echo "%"
	@echo "% $(APPNAME) is ready."
	@echo "%"

.PHONY: all clean header dist

clean:
	@rm -rf src/*.o *.dSYM $(APPNAME)



# For TEST
T_FLAGS	= -Iinclude -Isrc
T_LIBS	= -lcutil -lcurand -lgsl -lgslcblas
T_APPNAME = bin/test

T_CU	= $(wildcard t/*.cu)
T_CPP	= $(wildcard t/*.cpp)
T_OBJ	= $(addsuffix .o, $(basename $(T_CU))) $(addsuffix .o, $(basename $(T_CPP)))
C_OBJ	= $(filter-out %main.o, $(filter-out %kernel.o, $(OBJ)))


%.o: %.cu
	@echo Compiling: "$@ ( $< )"
	@$(CXX) $(T_FLAGS) -c -o $@ $<

%.o: %.cpp
	@echo Compiling: "$@ ( $< )"
	@$(CXX) $(T_FLAGS) -c -o $@ $<


test: $(T_OBJ) $(OBJ)
	@$(CXX) $(T_FLAGS) $(T_OBJ) $(C_OBJ) -o $(T_APPNAME) $(T_LIBS)

# eof
