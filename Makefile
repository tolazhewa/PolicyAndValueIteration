CC=g++
CFLAGS = --std=c++11
CLASSES = Environment.cpp
VIA_CLASSES = VIAgent.cpp
PIA_CLASSES = PIAgent.cpp

all: via pia

via: via.cpp
	$(CC) -o $@.o $< $(VIA_CLASSES) $(CLASSES) $(CFLAGS)

pia: pia.cpp
	$(CC) -o $@.o $< $(PIA_CLASSES) $(CLASSES) $(CFLAGS)

clean:
	rm *.o