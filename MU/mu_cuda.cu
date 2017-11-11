#include <bits/stdc++.h>

using namespace std;

#define TILE_WIDTH 16
#define TILE_DIM 32
#define BLOCK_ROWS 8

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
  		       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {

    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

__global__ void
transpose(float *odata, const float *idata, int rows, int cols)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  // int width = gridDim.x * TILE_DIM;
  int width = cols;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
  {
    if(y+j<rows && x<cols)
      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
    else
      tile[threadIdx.y+j][threadIdx.x] = 0.0;
  }


  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
  {
    if(y+j<rows && x<cols)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

__global__
void elewisemulti(float *A, float *B, float *C, int rows, int cols)
{
  int tx = threadIdx.x, ty = threadIdx.y,
      bx = blockIdx.x, by = blockIdx.y,
      row = by*blockDim.y+ty, col=blockDim.x*bx+tx;

  if(row<rows && col<cols)
    C[row][col] = A[row][col]*B[row][col];
}

__global__
void elewisediv(float *A, float *B, float *C, int rows, int cols)
{
  int tx = threadIdx.x, ty = threadIdx.y,
      bx = blockIdx.x, by = blockIdx.y,
      row = by*blockDim.y+ty, col=blockDim.x*bx+tx;

  if(row<rows && col<cols)
    C[row][col] = A[row][col]/B[row][col];
}

int main()
{
  int m,n;
  ifstream dataset("dataset.txt");
  float *Ahost,*Bhost,*Chost,*Bhost1,*Chost1;
  float *A,*B,*C;
  //cout << "Enter dimensions of the matrix : ";
  //cin>>m>>n;
  m = 943;
  n = 1682;
  cout << "Enter k value : ";
  int k;
  cin>>k;
  int epochs;
  cout<<"Enter epochs for training : ";
  cin>>epochs;

  Ahost = new float[m*n];
  Bhost = new float[m*k];
  Bhost1 = new float[m*k];
  Chost = new float[k*n];
  Chost1 = new float[k*n];

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  srand(time(NULL));

  memset(Ahost,0,m*n*sizeof(float));

  for(int i=0;i<100000;++i){
	int temp;
	dataset >> temp;
	int x,y;
	x = --temp;
	dataset >> temp;
	y = --temp;
	dataset >> temp;
	Ahost[x*n + y] = temp;
  }

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

  cudaMalloc((void **)&A, sizeof(float) * m*n);
  cudaMalloc((void **)&B, sizeof(float) * m*k);
  cudaMalloc((void **)&C, sizeof(float) * k*n);

  cudaMemcpy(A, Ahost, sizeof(float) * m*n, cudaMemcpyHostToDevice);
  cudaMemcpy(B, Bhost, sizeof(float) * m*k, cudaMemcpyHostToDevice);
  cudaMemcpy(C, Chost, sizeof(float) * k*n, cudaMemcpyHostToDevice);


  float *ACtranspose = new float[m*k];
  float *Ctranspose = new float[k*n];
  float *BACtranspose = new float[m*k];
  float *CCtranspose = new float[k*k];
  float *BCCtranspose = new float[m*k];
  cudaMalloc((void **)&ACtranspose, sizeof(float) * m*k);
  cudaMalloc((void **)&Ctranspose, sizeof(float) * k*n);
  cudaMalloc((void **)&CCtranspose, sizeof(float) * k*k);
  cudaMalloc((void **)&BCCtranspose, sizeof(float) * m*k);
  cudaMalloc((void **)&BACtranspose, sizeof(float) * m*k);

  dim3 gridSize1((k-1)/TILE_DIM + 1,(n-1)/TILE_DIM + 1,1);
  dim3 blockSize(TILE_DIM, BLOCK_ROWS, 1);

  dim3 gridSize2((m-1)/TILE_WIDTH + 1,(k-1)/TILE_WIDTH + 1,1);

  dim3 gridSize3((k-1)/TILE_WIDTH + 1,(k-1)/TILE_WIDTH + 1,1);


  float *BtransposeA = new float[k*n];
  float *Btranspose = new float[k*m];
  float *CBtransposeA = new float[k*n];
  float *BtransposeB = new float[k*k];
  float *BtransposeBC = new float[k*n];
  cudaMalloc((void **)&BtransposeA, sizeof(float) * k*n);
  cudaMalloc((void **)&Btranspose, sizeof(float) * k*m);
  cudaMalloc((void **)&CBtransposeA, sizeof(float) * k*n);
  cudaMalloc((void **)&BtransposeB, sizeof(float) * k*k);
  cudaMalloc((void **)&BtransposeBC, sizeof(float) * k*n);

  dim3 gridSize4((m-1)/TILE_DIM + 1,(k-1)/TILE_DIM + 1,1);

  dim3 gridSize5((k-1)/TILE_WIDTH + 1,(n-1)/TILE_WIDTH + 1,1);

  dim3 gridSize5((k-1)/TILE_WIDTH + 1,(k-1)/TILE_WIDTH + 1,1);

  for(int i=0;i<epochs;i++)
  {

    transpose<<<gridSize1, blockSize>>>(Ctranspose, C, k, n);
    matrixMultiply<<<gridSize2, blockSize>>>(A, Ctranspose, ACtranspose, m, n, n, k, m, k);
    elewisemulti<<<gridSize2, blockSize>>>(B, ACtranspose, BACtranspose, m, k);

    matrixMultiply<<<gridSize3, blockSize>>>(C, Ctranspose, CCtranspose, k, n, n, k, k, k);
    matrixMultiply<<<gridSize2, blockSize>>>(B, CCtranspose, BCCtranspose, m, k, k, k, m, k);
    elewisediv<<<gridSize2, blockSize>>>(BACtranspose, BCCtranspose, B, m, k);

    transpose<<<gridSize4, blockSize>>>(Btranspose, B, m, k);
    matrixMultiply<<<gridSize5, blockSize>>>(Btranspose, A, BtransposeA, k, m, m, n, k, n);
    elewisemulti<<<gridSize5, blockSize>>>(C, BtransposeA, CBtransposeA, k, n);

    matrixMultiply<<<gridSize6, blockSize>>>(Btranspose, B, BtransposeB, k, m, m, k, k, k);
    matrixMultiply<<<gridSize2, blockSize>>>(BtransposeB, C, BtransposeBC, k, k, k, n, k, n);
    elewisediv<<<gridSize2, blockSize>>>(CBtransposeA, BtransposeBC, C, k, n);

  }

  cudaMemcpy(Bhost, B, sizeof(float) * m*k, cudaMemcpyDeviceToHost);
  cudaMemcpy(Chost, C, sizeof(float) * k*n, cudaMemcpyDeviceToHost);

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

  cout<<"RMS error : "<<sqrt(sumError/(m*n))<<endl;

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
