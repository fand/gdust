# Makefile
# doc1: http://www.gnu.org/software/make/manual/make.html
# doc2: http://netbsd.gw.com/cgi-bin/man-cgi?make+1+NetBSD-current

CXX      = nvcc
CFLAGS   = -Iinclude -use_fast_math
# -O3
# -L../opt/boost/lib -I../opt/boost/include

LIBS     = -lcutil -lcurand -lgsl -lgslcblas

APPNAME  = gdustdtw

#####################################################################


SRC	= $(wildcard src/*.cu)
HDR	= $(wildcard include/*.hpp)
OBJ	= $(addsuffix .o, $(basename $(SRC)))


all:  header $(APPNAME) trailer


%.o: %.cu
	@echo Compiling: "$@ ( $< )"
	@$(CXX) $(CFLAGS) -c -o $@ $<

$(APPNAME): $(OBJ) 
	@echo Compiling: "$@ ( $^ )"
	@$(CXX)  $(CFLAGS) $(OBJ) -o $(APPNAME) $(LIBS)


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

# eof
