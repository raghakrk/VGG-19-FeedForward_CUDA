all:	kernel

kernel:	kernel.o
	nvcc -o kernel kernel.o

kernel.o:kernel.cu
	nvcc -c kernel.cu 

clean:
	rm *.o kernel
