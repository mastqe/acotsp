# Makefile for ACOTSP
VERSION=1.03

CC = gcc
NVCC = nvcc
OPTIM_FLAGS = -O3
WARN_FLAGS = -Wall -std=c99 -pedantic -Wno-unused-result
# PROF_FLAGS = -pg -fprofile-arcs -ftest-coverage
# DEBUG_FLAGS = -g
PAR_FLAGS = -fopenmp
CUDA_FLAGS = -lcudart
CFLAGS = $(WARN_FLAGS) $(OPTIM_FLAGS) $(PROF_FLAGS) $(DEBUG_FLAGS)

LDLIBS = -lm

# To change the default timer implementation, uncomment the line below
# or call 'make TIMER=unix'
#TIMER = dos
TIMER = unix

all: clean acotsp omp_acotsp cuda_acotsp

clean:
	@$(RM) ./out/* acotsp omp_acotsp cuda_acotsp

acotsp: ./out/acotsp.o ./out/TSP.o ./out/ants.o ./out/InOut.o ./out/utilities.o ./out/ls.o ./out/parse.o ./out/$(TIMER)_timer.o
	$(CC) $(CFLAGS) -o acotsp $^ $(LDLIBS)

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

# OpenMP section - only certain files need special compilation
omp_acotsp: ./out/omp_acotsp.o ./out/TSP.o ./out/omp_ants.o ./out/InOut.o ./out/utilities.o ./out/ls.o ./out/parse.o ./out/$(TIMER)_timer.o
	$(CC) $(CFLAGS) $(PAR_FLAGS) -o omp_acotsp $^ $(LDLIBS)

./out/omp_acotsp.o: ./src/acotsp.c
	$(CC) $(CFLAGS) $(PAR_FLAGS) -c -o $@ $<

./out/omp_ants.o: ./src/ants.c ./src/ants.h
	$(CC) $(CFLAGS) $(PAR_FLAGS) -c -o $@ $<


# CUDA Section - for simplicity all code is copied and compiled 
# 			independently from CPU versions
cuda_acotsp: ./out/cuda_acotsp.o ./out/cuda_TSP.o ./out/cuda_ants.o ./out/cuda_InOut.o ./out/cuda_utilities.o ./out/cuda_ls.o ./out/cuda_parse.o ./out/cuda_$(TIMER)_timer.o
	$(NVCC) $(CUDA_FLAGS) -o cuda_acotsp $^ $(LDLIBS)

./out/cuda_acotsp.o: ./cuda_src/acotsp.cu
	$(NVCC) $(CUDA_FLAGS) -c -o $@ $<

./out/cuda_TSP.o: ./src/TSP.c ./src/TSP.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/cuda_ants.o: ./src/ants.cu ./src/ants.h
	$(NVCC) $(CUDA_FLAGS) -c -o $@ $<

./out/cuda_InOut.o: ./src/InOut.c ./src/InOut.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/cuda_utilities.o: ./src/utilities.c ./src/utilities.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/cuda_ls.o: ./src/ls.c ./src/ls.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/cuda_parse.o: ./src/parse.c ./src/parse.h
	$(CC) $(CFLAGS) -c -o $@ $<

./out/cuda_$(TIMER)_timer.o: ./src/$(TIMER)_timer.c ./src/timer.h
	$(CC) $(CFLAGS) -c -o $@ $<
