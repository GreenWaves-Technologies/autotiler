TILER_VER=3.0.0
TILER_LIB=libtile.${TILER_VER}.a
ifdef GAP_SDK_HOME
export TILER_URL=$(GAP_SDK_HOME)/.tiler_url
else
export TILER_URL=.tiler_url
endif

all: lib/libtile.a

clean:
	rm -rf lib/libtile*
	rm -f $(TILER_URL)

ifeq (,$(wildcard $(TILER_URL)))
$(TILER_URL): get_tiler.py
	python3 get_tiler.py
endif

lib/libtile.a: $(TILER_URL)
	mkdir -p lib
	rm -rf lib/libtile*
	echo ${TILER_LIB} | wget --no-use-server-timestamps --base=`cat $(TILER_URL)` --input-file=- -O $@

.PHONY: all clean
