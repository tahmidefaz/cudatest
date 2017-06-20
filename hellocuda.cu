#include <iostream>

__global__ void add(int *a, int *b, int *c){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  //if (index < n)
  c[index] = a[index] + b[index];
}

void random_ints(int *p, int s){
  for(int i=0; i < s; i++){
    p[i] = rand();
  }
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512
int main(){
  int *a, *b, *c;  // host copies
  int *d_a, *d_b, *d_c;  //device copies
  int size = N * sizeof(int);
  
  // Space allocation for device copies
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);
  
  // Alloc space for host copies of a, b, c and setup input values
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size); random_ints(b, N);
  c = (int *)malloc(size);
  
  // Copy inputs to Device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  
  // Launch add Kernel on GPU with N threads
  add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
  //add<<<(N + M-1) / M,M>>>(d_a, d_b, d_c, N);
  
  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  
  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  
  std::cout<<"Done!"<<std::endl;
  
  return 0;
}