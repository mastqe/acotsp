# Makefile for ACOTSP
VERSION=1.03

CC = gcc
OPTIM_FLAGS = -O1
WARN_FLAGS = -Wall -std=c99 -pedantic -Wno-unused-result
# PROF_FLAGS = -pg -fprofile-arcs -ftest-coverage
PAR_FLAGS = #-fopenmp
CFLAGS = $(WARN_FLAGS) $(OPTIM_FLAGS) $(PROF_FLAGS) $(PAR_FLAGS)

LDLIBS = -lm

# To change the default timer implementation, uncomment the line below
# or call 'make TIMER=unix'
#TIMER = dos
TIMER = unix

OUT_FILES = $(wildcard ./out/*.o)

all: clean acotsp

clean:
	@$(RM) ./out/* acotsp

acotsp: ./out/acotsp.o ./out/TSP.o ./out/ants.o ./out/InOut.o ./out/utilities.o ./out/ls.o ./out/parse.o ./out/$(TIMER)_timer.o
	$(CC) $(CFLAGS) -o acotsp $(OUT_FILES) $(LDLIBS) 

./out/acotsp.o: ./src/acotsp.c
	$(CC) $(CFLAGS) -c -o $@ $<

./out/TSP.o: ./src/TSP.c ./src/TSP.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/ants.o: ./src/ants.c ./src/ants.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/InOut.o: ./src/InOut.c ./src/InOut.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/utilities.o: ./src/utilities.c ./src/utilities.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/ls.o: ./src/ls.c ./src/ls.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/parse.o: ./src/parse.c ./src/parse.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/$(TIMER)_timer.o: ./src/$(TIMER)_timer.c ./src/timer.h
	$(CC) $(CFLAGS) -c -o $@ $<
