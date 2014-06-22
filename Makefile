CC=g++
CFLAGS=-W -Wall -Wextra -pedantic -O2
LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lm
PROJ=segment

all: $(PROJ)

$(PROJ): main.o threshold.o
	$(CC) -o $@ $^ $(LIBS)

main.o: main.cc threshold.h
	$(CC) $(CFLAGS) -c -o $@ $<

threshold.o: threshold.cc threshold.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o $(PROJ)

