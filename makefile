CC=nvcc
CFLAGS=-std=c++11

all: 
	$(CC) $(CFLAGS) main.cu -o multi_gpu_test -lm
