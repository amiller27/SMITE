libmetis:
	cd METIS/build && make -j16

HEADERS = -Ieigen \
	-Ifmt/include -DFMT_HEADER_ONLY

LMETIS = -lmetis -LMETIS/build/libmetis

FLAGS = $(HEADERS) \
        -IMETIS/include $(LMETIS) \
        -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=64 \
        -IMETIS/GKlib -DGKRAND_SAVE_TRACE

lib: test.cc libmetis
	g++ test.cc $(FLAGS) -c -fPIC -shared -o libmetis_test.so

main: main.cc lib
	g++ main.cc -lmetis_test -L. $(FLAGS) 

run: lib main
    LD_LIBRARY_PATH=METIS/build/libmetis ./a.out