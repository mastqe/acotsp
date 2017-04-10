# Makefile for ACOTSP
VERSION=1.03

CC = gcc
OPTIM_FLAGS = -O3
WARN_FLAGS = -Wall -std=c99 -pedantic -Wno-unused-result
# PROF_FLAGS = -pg -fprofile-arcs -ftest-coverage
PAR_FLAGS = -fopenmp
DEBUG_FLAGS = #-g
CFLAGS = $(WARN_FLAGS) $(OPTIM_FLAGS) $(PROF_FLAGS) $(DEBUG_FLAGS)

LDLIBS = -lm

# To change the default timer implementation, uncomment the line below
# or call 'make TIMER=unix'
#TIMER = dos
TIMER = unix

all: clean acotsp omp_acotsp

clean:
	@$(RM) ./out/* acotsp omp_acotsp

acotsp: ./out/acotsp.o ./out/TSP.o ./out/ants.o ./out/InOut.o ./out/utilities.o ./out/ls.o ./out/parse.o ./out/$(TIMER)_timer.o
	$(CC) $(CFLAGS) -o acotsp $^ $(LDLIBS)

omp_acotsp: ./out/omp_acotsp.o ./out/TSP.o ./out/omp_ants.o ./out/InOut.o ./out/utilities.o ./out/ls.o ./out/parse.o ./out/$(TIMER)_timer.o
	$(CC) $(CFLAGS) $(PAR_FLAGS) -o omp_acotsp $^ $(LDLIBS)

./out/acotsp.o: ./src/acotsp.c
	$(CC) $(CFLAGS) -c -o $@ $<

./out/omp_acotsp.o: ./src/acotsp.c
	$(CC) $(CFLAGS) $(PAR_FLAGS) -c -o $@ $<

./out/TSP.o: ./src/TSP.c ./src/TSP.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/ants.o: ./src/ants.c ./src/ants.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/omp_ants.o: ./src/ants.c ./src/ants.h
	$(CC) $(CFLAGS) $(PAR_FLAGS) -c -o $@ $<

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
