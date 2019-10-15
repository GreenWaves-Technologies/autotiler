TILER_VER=3.0.0
TILER_LIB=libtile.${TILER_VER}.a

all: lib/libtile.a

clean:
	rm -rf lib/libtile*
	rm -f .tiler_url

ifeq (,$(wildcard .tiler_url))
.tiler_url: get_tiler.py
	python3 get_tiler.py
endif

lib/libtile.a: .tiler_url
	mkdir -p lib
	rm -rf lib/libtile*
	echo ${TILER_LIB} | wget --no-use-server-timestamps --base=`cat .tiler_url` --input-file=- -O $@

.PHONY: all clean
