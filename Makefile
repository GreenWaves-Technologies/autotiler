TILER_VER=2.1.0
TILER_LIB=libtile.${TILER_VER}.a

all: lib/libtile.a

clean:
	rm -rf lib/libtile.a

ifeq (,$(wildcard .tiler_url))
.tiler_url: get_tiler.py
	python3 get_tiler.py
endif

lib:
	mkdir lib

lib/${TILER_LIB}: lib .tiler_url
	rm -rf lib/libtile.a
	echo ${TILER_LIB} | wget --no-use-server-timestamps --base=`cat .tiler_url` --input-file=- -O $@

lib/libtile.a: lib/${TILER_LIB}
	ln -s ${TILER_LIB} $@

.PHONY: all clean
