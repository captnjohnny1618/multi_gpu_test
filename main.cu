// This program is a proof of concept for threading the reconstruction
// to the GPUs as separate short-stack reconstructions and then stitching
// after all threads have been completed.

// If the code behaves as it should you should observe some stuff being printed
// as the code initializes (this will likely come out of order since the code is)
// threaded and then the numbers 0.0-100.0 printed out in increments of 1.0.  These
// should be *in order* although you may observe that the last values are 0.0 when it should
// be 100.  This is not reflective of the method, but just a blocking issue that's too minor
// to bother fixing.

// To make this run on your machine it is likely that paths will need to be modified.

// Compile on linux with:
//      nvcc main.cu -std=c++11 -o multi_gpu_test -lm

// Compile in visual studio by adding main.cu to a cuda project and making sure that the c++
// math library is being linked. Code required c++11 for threading.

#include <string.h>
#include <math.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>

#include <cuda_runtime.h>

__global__ void recon_kernel(float * d_array,float start, float slice_thickness){
    d_array[threadIdx.x] = threadIdx.x*slice_thickness + start;
}

void dummy_recon_cpu(float start, float end, float slice_thickness,std::string output_filepath){

    int n_slices = (end - start)/slice_thickness + 1;
    float * array = new float[n_slices];

    printf("Block start: %.2f\n",start);
    printf("Block end: %.2f\n"  ,end);
    printf("Block slices: %d\n" ,n_slices);
    
    for (int i=0;i<n_slices;i++){
        array[i] = i*slice_thickness + start;
        //printf("%.02f\n",array[i]);
    }

    std::cout << output_filepath << std::endl;
    std::ofstream outfile(output_filepath,std::ios::binary | std::ios::out);
    outfile.write((char *)&array[0],n_slices*sizeof(float));
    outfile.close();
    
    delete[] array;
}

void dummy_recon_gpu(float start, float end, float slice_thickness,std::string output_filepath,int device_idx){
    
    printf("Block GPU: %d\n",device_idx);
    //printf("Block start: %.2f\n",start);
    //printf("Block end: %.2f\n"  ,end);
    //printf("Block slices: %d\n" ,n_slices);

    cudaSetDevice(device_idx);

    int n_slices = (end - start)/slice_thickness + 1;
    float * h_array = new float[n_slices];
    float * d_array;
    cudaMalloc(&d_array,n_slices*sizeof(float));

    recon_kernel<<<1,n_slices>>>(d_array,start,slice_thickness);
    cudaMemcpy(h_array,d_array,n_slices*sizeof(float),cudaMemcpyDeviceToHost);
    
    std::cout << output_filepath << std::endl;
    std::ofstream outfile(output_filepath,std::ios::binary | std::ios::out);
    outfile.write((char *)&h_array[0],n_slices*sizeof(float));
    outfile.close();

    cudaFree(d_array);
    delete[] h_array;

}

int main(int argc, char ** argv){
    // ********************
    // Initial config stuff
    // ********************    
    std::string output_filepath = "/home/john/Code/multigpu_test/test.bin";
    
    int n_devices;
    cudaGetDeviceCount(&n_devices);
    std::cout << "Cuda Devices Found: " << n_devices << std::endl;

    float start = 0.0f;
    float end   = 100.0f;
    float slice_thickness = 1.0f;
    int n_slices = (end-start)/slice_thickness + 1;

    // ********************    
    // Dispatch threads to do recon in blocks
    // ********************    
    // configure block metadata
    float block_n_slices = ceil(n_slices / n_devices);
    float block_start = start;
    float block_end = (block_n_slices-1)*slice_thickness + block_start;
    
    std::vector<std::thread> threads;    
    for (int i=0;i<n_devices;i++){
        std::string filename = "/tmp/recon_";

        printf("===========================\n");
        printf("Block %d\n",i);
        printf("===========================\n");

        //threads.emplace_back(dummy_recon_cpu,block_start,block_end,slice_thickness,filename + std::to_string(i) + ".bin");
        threads.emplace_back(dummy_recon_gpu,block_start,block_end,slice_thickness,filename + std::to_string(i) + ".bin",i);        
        block_start = block_end + slice_thickness;
        block_end   = std::min((block_n_slices-1)*slice_thickness + block_start,end);        
    }

    // Wait for threads to finish
    for (int i=0;i<n_devices;i++){
        threads[i].join();
    }

    std::cout << "ALL THREADS COMPLETE" << std::endl;
    std::cout << "Showing reassembled slice locations" << std::endl;
    
    // ********************
    // Reassemble all files and move to a final location
    // ********************    
    std::ofstream final_recon(output_filepath.c_str(),std::ios::binary | std::ios::out);
    for (int i=0;i<n_devices;i++){
        
        std::string filename = "/tmp/recon_" + std::to_string(i) + ".bin";
        std::ifstream tmp(filename,std::ios::binary | std::ios::in);
        if (!tmp.good())
            std::cout << "Could not find file: " << filename << std::endl;
        final_recon << tmp.rdbuf();
        tmp.close();
    }
    final_recon.close();

    // ********************    
    // Read the final file and print
    // ********************    
    float *tmp_array = new float[n_slices];
    std::ifstream final_recon_in(output_filepath.c_str(),std::ios::binary);
    final_recon_in.read((char*)&tmp_array[0],n_slices*sizeof(float));
    for (int i=0; i<n_slices; i++){
        printf("%.02f\n",tmp_array[i]);
    }

    delete[] tmp_array;

    return 0;    
}


    
