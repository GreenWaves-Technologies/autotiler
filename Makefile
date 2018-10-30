TILER_VER=1.0.0
TILER_LIB=libtile.${TILER_VER}.a

all: lib/libtile.a

clean:
	rm lib/libtile.a

.tiler_url: get_tiler.py
	python3 get_tiler.py

lib:
	mkdir lib

lib/${TILER_LIB}: lib .tiler_url
	echo ${TILER_LIB} | wget --base=`cat .tiler_url` --input-file=- -O $@ 

lib/libtile.a: lib/${TILER_LIB}
	ln -s $< $@

.PHONY: all clean