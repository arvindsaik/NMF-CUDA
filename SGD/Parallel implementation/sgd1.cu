#include <bits/stdc++.h>

using namespace std;

void sgd_CPU(float *A,float *B,float *C,int epochs,float lamda,float alpha,int m,int n,int k){
    for(int i=0;i<epochs;++i){
      for(int x = 0;x<m;++x){
        for(int y = 0;y<n;++y){
          float error = 0;
          float temp = 0;
          for(int iter = 0;iter < k;++iter){
            temp += B[x*k + iter]*C[iter*n + y];
          }
          error = A[x*n+y] - temp;
          for(int iter = 0;iter < k;++iter){
            B[x*k + iter] = B[x*k + iter] + alpha*((error * C[iter*n + y]) - lamda*(B[x*k + iter]));
            C[iter*n + y] = C[iter*n + y] + alpha*((error * B[x*k + iter]) - lamda*(C[iter*n + y]));
          }
        }
      }
    }
}

__global__ void sgd_kernel(float *A,float *B,float *C,int epochs,float lamda,float alpha,int m,int n,int k){
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if(y<n && x<m){
    for(int i=0;i<epochs;++i){
      float error = 0;
      float temp = 0;
      for(int iter = 0;iter < k;++iter){
        temp += B[x*k + iter]*C[iter*n + y];
      }
      // cout<<x<<" "<<y<<" "<<temp<<endl;
      error = A[x*n+y] - temp;
      for(int iter = 0;iter < k;++iter){
        atomicAdd(&B[x*k + iter],alpha*((error * C[iter*n + y]) - lamda*(B[x*k + iter])));
        atomicAdd(&C[iter*n + y],alpha*((error * B[x*k + iter]) - lamda*(C[iter*n + y])));
        // B[x*k + iter] = B[x*k + iter] + alpha*((error * C[iter*n + y]) - lamda*(B[x*k + iter]));
        // C[iter*n + y] = C[iter*n + y] + alpha*((error * B[x*k + iter]) - lamda*(C[iter*n + y]));
      }
    }
  }
}

int main(){
  int m,n;
  float *Ahost,*Bhost,*Chost,*Bhost1,*Chost1;
  float *A,*B,*C;
  cout << "Enter dimensions of the matrix : ";
  cin>>m>>n;
  cout << "Enter k value : ";
  int k;
  cin>>k;
  int epochs;
  cout<<"Enter epochs for training : ";
  cin>>epochs;
  float alpha;
  cout<<"Enter alpha for training : ";
  cin>>alpha;
  float lamda;
  cout<<"Enter lamda (regularisation variable) for training : ";
  cin>>lamda;
  Ahost = new float[m*n];
  Bhost = new float[m*k];
  Bhost1 = new float[m*k];
  Chost = new float[k*n];
  Chost1 = new float[k*n];
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  srand(time(NULL));

  //randomising the input array A
  for(int i=0;i<m;++i){
    for(int j=0;j<n;++j){
      Ahost[i*n + j] = rand()%100 + 1.03;
    }
  }

  //randomising the output array B
  for(int i=0;i<m;++i){
    for(int j=0;j<k;++j){
      Bhost[i*k + j] = rand()%2;
      Bhost1[i*k + j] = Bhost[i*k + j];
    }
  }

  //randomising the output array C
  for(int i=0;i<k;++i){
    for(int j=0;j<n;++j){
      Chost[i*n + j] = rand()%2;
      Chost1[i*n + j] = Chost[i*n + j];
    }
  }

  // cout<<" A : "<<endl;
  //
  // for(int i=0;i<m;++i){
  //   for(int j=0;j<n;++j){
  //     cout<<Ahost[i*n+j]<<" ";
  //   }
  //   cout<<endl;
  // }

  // cout<<" B : \n";
  //
  // for(int i=0;i<m;++i){
  //   for(int j=0;j<k;++j){
  //     cout<<Bhost[i*k+j]<< " ";
  //   }
  //   cout<<endl;
  // }

  cudaMalloc((void **)&A, sizeof(float) * m*n);
  cudaMalloc((void **)&B, sizeof(float) * m*k);
  cudaMalloc((void **)&C, sizeof(float) * k*n);

  cudaMemcpy(A, Ahost, sizeof(float) * m*n, cudaMemcpyHostToDevice);
  cudaMemcpy(B, Bhost, sizeof(float) * m*k, cudaMemcpyHostToDevice);
  cudaMemcpy(C, Chost, sizeof(float) * k*n, cudaMemcpyHostToDevice);


  dim3 gridSize((n-1)/32 + 1,(m-1)/32 + 1,1);
  dim3 blockSize(32,32,1);

  cudaEventRecord(start);

  sgd_kernel<<<gridSize,blockSize>>>(A,B,C,epochs,lamda,alpha,m,n,k);

  cudaEventRecord(stop);


  cudaMemcpy(Bhost, B, sizeof(float) * m*k, cudaMemcpyDeviceToHost);
  cudaMemcpy(Chost, C, sizeof(float) * k*n, cudaMemcpyDeviceToHost);

  // cout<<" Product : \n";
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cout<<"GPU kernel took "<<milliseconds<<" milliseconds"<<endl;

  float temp[m][n];

  for(int i=0;i<m;++i){
    for(int j=0;j<n;++j){
      temp[i][j] = 0;
      for(int l=0;l<k;++l){
        temp[i][j]+= Bhost[i*k+l]*Chost[l*n+j];
      }
      // cout<<temp[i][j]<< " ";
    }
    // cout<<endl;
  }
  float sumError = 0;
  for(int i=0;i<m;++i){
    for(int j=0;j<n;++j){
      float errorMat  = Ahost[i*n+j] - temp[i][j];
      sumError += errorMat*errorMat;
    }
  }

  cout<<"RMS error : "<<sqrt(sumError)<<endl;

  double s = clock();
  sgd_CPU(Ahost,Bhost1,Chost1,epochs,lamda,alpha,m,n,k);
  double e = clock();

  cout<<"CPU implementation took "<< 1000*(e-s)/CLOCKS_PER_SEC<<" milliseconds\n";

  return 0;
}
