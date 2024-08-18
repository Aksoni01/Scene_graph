
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

typedef long long ll;

__global__ void convolution(long int* matrix,long int* filter,long int* answer,int k,int m,int n){
    extern __shared__ long int temp[];
    ll id = blockDim.x * blockIdx.x + threadIdx.x;

    int chunk_size=ceil((1.0*k*k)/n);
    for(int i=0;i<chunk_size;i++){
       int index=threadIdx.x+i*n;
      if(index<k*k){
        temp[index]=filter[index];
      }
    }

    __syncthreads();

    ll start_row=(id/n)-(k/2),start_col=(id%n)-(k/2),ans=0;
    for(int i=0;i<k;i++){
      for(int j=0;j<k;j++){
        if(i+start_row>=0 && start_row+i<m && j+start_col>=0 && j+start_col<n)
           ans+=matrix[(start_row+i)*n+(start_col+j)]*temp[i*k+j];
      }
    }
    answer[id]=ans;
}


int main(int argc, char** argv) {
    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];

    for (ll i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }
    for (ll i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

   

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
    **/

    /****************************************************Start Here***********************************************************/

    long int* matrix;
    long int* filter;
    long int* answer;

    cudaMalloc(&matrix,m*n*sizeof(long int));
    cudaMalloc(&filter,k*k*sizeof(long int));
    cudaMalloc(&answer, m * n * sizeof(long int));

    cudaMemcpy(matrix,h_mat,m*n*sizeof(long int),cudaMemcpyHostToDevice);
    cudaMemcpy(filter,h_filter,k*k*sizeof(long int),cudaMemcpyHostToDevice);



    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch

    convolution<<<m,n,k*k*sizeof(long int)>>>(matrix,filter,answer,k,m,n);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    cudaMemcpy(h_ans, answer, m * n * sizeof(long int), cudaMemcpyDeviceToHost);

    cudaFree(matrix);
    cudaFree(filter);
    cudaFree(answer);


    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
    */



    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}
