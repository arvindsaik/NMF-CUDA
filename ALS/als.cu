
#define SURPASS_NAN
// #define USE_CG
//if cojugate gradient solver generates results in FP16
//#define CUMF_TT_FP16
//#define CUMF_XX_FP16
#define CG_ITER 6
//#define CUMF_SAVE_MODEL
#include "als.h"
//#include "device_utilities.h"
#include "host_utilities.h"
#include <fstream>
#include <assert.h>
#include<stdlib.h>
#include<stdio.h>
#include <string>
#define SCAN_BATCH 28
#include <iostream>
using namespace std;

void saveDeviceFloatArrayToFile(string fileName, int size, float* d_array){
	float* h_array;
	cudacall(cudaMallocHost( (void** ) &h_array, size * sizeof(h_array[0])) );
	cudacall(cudaMemcpy(h_array, d_array, size * sizeof(h_array[0]),cudaMemcpyDeviceToHost));
	FILE * outfile = fopen(fileName.c_str(), "wb");
	fwrite(h_array, sizeof(float), size, outfile);
	fclose(outfile);
	cudaFreeHost(h_array);
}
int updateX(const int batch_size, const int batch_offset, float * ythetaT, float * tt, float * XT,
		cublasHandle_t handle, const int m, const int n, const int f, const int nnz,
		float** devPtrTTHost, float **devPtrYthetaTHost)
    {
      float **devPtrTT = 0;
    	int *INFO;
    	for (int k = 0; k < batch_size; k++) {
    		devPtrTTHost[k] = &tt[k * f * f];
    	}
    	cudacall(cudaMalloc((void** ) &devPtrTT, batch_size * sizeof(*devPtrTT)));
    	cudacall(cudaMemcpy(devPtrTT, devPtrTTHost, batch_size * sizeof(*devPtrTT),cudaMemcpyHostToDevice));
    	//cudacall( cudaMalloc(&P, f * batch_size * sizeof(int)) );
    	cudacall( cudaMalloc(&INFO, batch_size * sizeof(int) ));

      //Performing matrix inversion
    	cublascall(cublasSgetrfBatched(handle, f, devPtrTT, f, NULL, INFO, batch_size));

    	cudaThreadSynchronize();

      float **devPtrYthetaT = 0;

    	for (int k = 0; k < batch_size; k++) {
    		devPtrYthetaTHost[k] = &ythetaT[batch_offset * f + k * f];
    	}
    	cudacall(cudaMalloc((void** ) &devPtrYthetaT, batch_size * sizeof(*devPtrYthetaT)));
    	cudacall(cudaMemcpy(devPtrYthetaT, devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaT), cudaMemcpyHostToDevice));

    	int * info2 = (int *) malloc(sizeof(int));
    	cublascall( cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
    			(const float ** ) devPtrTT, f, NULL, devPtrYthetaT, f, info2, batch_size) );

    	cudaThreadSynchronize();
    	cudaError_t cudaStat1 = cudaGetLastError();
    	if (cudaStat1 != cudaSuccess) {
    		fprintf(stderr,"Failed to launch cublasSgetrsBatched (error code: %s)!\n", cudaGetErrorString(cudaStat1));
    		exit(EXIT_FAILURE);
    	}

    	cudacall( cudaMemcpy(&XT[batch_offset * f], &ythetaT[batch_offset * f],
    			batch_size * f * sizeof(float), cudaMemcpyDeviceToDevice) );

      cudacall(cudaFree(devPtrTT));
    	//cudacall(cudaFree(P));
    	cudacall(cudaFree(INFO));
    	cudacall(cudaFree(devPtrYthetaT));
    	return 0;

    }

		int updateTheta(const int batch_size, const int batch_offset, float * xx, float * yTXT, float * thetaT,
				cublasHandle_t handle, const int m, const int n, const int f, const int nnz,float ** devPtrXXHost, float **devPtrYTXTHost )
		    {
		      float **devPtrXX = 0;

		    	for (int k = 0; k < batch_size; k++) {
		    		devPtrXXHost[k] = &xx[k * f * f];
		    	}
		    	cudacall(cudaMalloc((void** ) &devPtrXX, batch_size * sizeof(*devPtrXX)));
		    	cudacall(cudaMemcpy(devPtrXX, devPtrXXHost, batch_size * sizeof(*devPtrXX), cudaMemcpyHostToDevice));

		      int *INFO;

		    	cudacall(cudaMalloc(&INFO, batch_size * sizeof(int)));
		    	cublascall(cublasSgetrfBatched(handle, f, devPtrXX, f, NULL, INFO, batch_size));
		    	cudaThreadSynchronize();

		      float **devPtrYTXT = 0;

		    	for (int k = 0; k < batch_size; k++) {
		    		devPtrYTXTHost[k] = &yTXT[batch_offset * f + k * f];
		    	}

		      cudacall(cudaMalloc((void** ) &devPtrYTXT, batch_size * sizeof(*devPtrYTXT)));
		    	cudacall(cudaMemcpy(devPtrYTXT, devPtrYTXTHost, batch_size * sizeof(*devPtrYTXT),cudaMemcpyHostToDevice));

		    	int * info2 = (int *) malloc(sizeof(int));
		    	cublascall( cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
		    			(const float ** ) devPtrXX, f, NULL, devPtrYTXT, f, info2, batch_size) );
		    	cudaThreadSynchronize();
		    	cudaError_t cudaStat1 = cudaGetLastError();
		    	if (cudaStat1 != cudaSuccess) {
		    		fprintf(stderr,"Failed to launch cublasSgetrsBatched (error code: %s)!\n", cudaGetErrorString(cudaStat1));
		    		exit(EXIT_FAILURE);
		    	}

		    	cudacall( cudaMemcpy( &thetaT[batch_offset * f], &yTXT[batch_offset * f],
		    	                        batch_size * f * sizeof(float), cudaMemcpyDeviceToDevice) );

		      cudaFree(devPtrXX);
		    	cudaFree(INFO);
		    	free(info2);
		    	cudaFree(devPtrYTXT);
		    	return 0;

		    }

				__global__
				void RMSE(const float * csrVal, const int* cooRowIndex, const int* csrColIndex, const float * __restrict__ thetaT,
				    const float * __restrict__ XT, float * error, const int nnz, const int error_size, const int f)
				    {
				      int i = blockIdx.x * blockDim.x + threadIdx.x;
				      if(i<nnz)
				      {
				        int row = cooRowIndex[i];
				        int col = csrColIndex[i];
				        float e = csrVal[i];

				        for(int k=0;k<f;k++)
				        {
				          #ifdef SURPASS_NAN
				    			//a and b could be; there are user/item in testing but not training set
				    			float a = thetaT[f * col + k];
				    			float b = XT[f * row + k];
				    			//if(isnan(a)||isnan(b))//nan not working in some platform
				    			if(a!=a||b!=b)
				    				break;
				    			else
				    				e -= a * b;
				    			//if(isnan(a)) printf("row: %d, col: %d\n", row, col);
				    			//if(isnan(b)) printf("b[%d]: %f.\n", i, b);
				    			#else
				    			e -= thetaT[f * col + k] * XT[f * row + k];
				    			#endif
				        }
				        atomicAdd(&error[i%error_size], e*e);
				      }
				    }



/*a generic kernel to get the hermitian matrices
 * as the left-hand side of the equations, to update X in ALS
 *examplary F = 100, T = 10
 */
 __global__
 void get_hermitianT10(const int batch_offset, float *tt,
                     const int *csrRowIndex, const int *csrColIndex, const float lambda,
                   const int m, const int F, const float * __restrict__ thetaT)
 {
   extern __shared__ float2 thetaTemp[];
   // Each row of Rating has a thread block and the offset is the Number of rows aldready computed
   int row = blockIdx.x + batch_offset;
   if(row < m)
   {
     int start  = csrRowIndex[row];
     int end = csrRowIndex[row+1];

     // Number of non zero elements in the ith row of the Rating Matrix
     //SCAN_BATCH is probalbly the bin size ie. to be loaded in the shared memeory
     int iterations = (end-start+1)/SCAN_BATCH + 1;

     float temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
 		float temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
 		float temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
 		float temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
 		float temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
 		float temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
 		float temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
 		float temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
 		float temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
 		float temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;

     int N = F/T10;
     int effective_block_size = N*(N+1)/2;

     int tile_x = 0;
     int tile_y = 0;

     for(int i=0;i<N;i++)
     {
       int end = ((2*N-1)*(i+1))/2;
       if(threadIdx.x<end)
       {
         tile_x = i*T10;
         tile_y = (N + threadIdx.x - end) * T10;
         break;
       }
     }

     int index = blockIdx.x*F*F;


     for(int iter=0;iter<iterations;iter++)
     {
       //Phase 1: Copying from Global Memory to shared Memory
       if(threadIdx.x<F/2)
       {
         for(int k=0;k<SCAN_BATCH;k++)
         {
           if(iter*SCAN_BATCH+k<end-start) //Border Condition
           {
             float2 theta;
             //Stored in column majour order
 						theta.x = thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x];
 						theta.y = thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x+1];
 						thetaTemp[k * F/2 + threadIdx.x] = theta;
           }
           //not enough theta to copy, set zero
 					else
 						memset(&thetaTemp[k*F/2 + threadIdx.x], 0, 2*sizeof(float));
         }
       }

       __syncthreads();

       //Phase 2: calculating A and storing in register
       //tid = 0 calculates the first 10 in thetaTemp and thetaTempTrans, tid = 1 calculates first 10 in thetaTemp and 10-20 in thetaTempTrans...
       // so total threads needed is 10(for the first 10 elements in thetaTemp with every set of 10 elements in thetaTempTrans)+
       // 9(second 10 elements in thetaTemp with set 1-9 elements thetaTempTrans)
       // + 8 + .. + 1 = 55 we are not using 100 threads as there are multilpe duplicate calculations
       if(threadIdx.x < effective_block_size)
       {
         for(int k = 0; k < SCAN_BATCH; k++){
 					accumulate_in_registers();
 				}
       }
     }
     //end of iteration in copying from smem and aggregating in register
 		__syncthreads();

     //Phase 3: Copying from registers to global memory
     		if(threadIdx.x < effective_block_size)
         {
           fill_lower_half_from_registers();

           if(tile_x != tile_y)
           {
 				        fill_upper_half_from_registers();
 			    }

           if(tile_x == tile_y)
           {
 				        for(int k = 0; k < T10; k++)
 					           tt[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
 			    }
         }
   }
 }

float doALS(const int* csrRowIndexHostPtr, const int* csrColIndexHostPtr, const float* csrValHostPtr,
		const int* cscRowIndexHostPtr, const int* cscColIndexHostPtr, const float* cscValHostPtr,
		const int* cooRowIndexHostPtr, float* thetaTHost, float* XTHost,
		const int * cooRowIndexTestHostPtr, const int * cooColIndexTestHostPtr, const float * cooValHostTestPtr,
		const int m, const int n, const int f, const long nnz, const long nnz_test, const float lambda,
		const int ITERS, const int X_BATCH, const int THETA_BATCH, const int DEVICEID)
{
	cudaSetDevice(DEVICEID);
	printf("*******parameters: m: %d, n:  %d, f: %d, nnz: %ld \n", m, n, f, nnz);
	//Creating device pointers
	int * csrRowIndex = 0;
	int * csrColIndex = 0;
	float * csrVal = 0;
	float * thetaT = 0;
	float * tt = 0;
	float * XT = 0;
	float * cscVal =0;
	int * cscRowIndex = 0;
	int * cscColIndex = 0;

	//Coo format is used for calculating the root mean square error
	int * cooRowIndex =0;
	float * cooVal_test;
	int * cooRowIndex_test;
	int * cooColIndex_test;
	float final_rmse = 0;
	//Allocating memeory to the device pointers

	cudacall(cudaMalloc((void** ) &cscRowIndex,nnz * sizeof(cscRowIndex[0])));
	cudacall(cudaMalloc((void** ) &cscColIndex, (n+1) * sizeof(cscColIndex[0])));
	cudacall(cudaMalloc((void** ) &cscVal, nnz * sizeof(cscVal[0])));

	//thetaT : f * N
	cudacall(cudaMalloc((void** ) &thetaT, f * n * sizeof(thetaT[0])));

	//X : M * f
	cudacall(cudaMalloc((void** ) &XT, f * m * sizeof(XT[0])));

	//Copying data from host to device

	cudacall(cudaMemcpy(cscRowIndex, cscRowIndexHostPtr,(size_t ) nnz * sizeof(cscRowIndex[0]), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(cscColIndex, cscColIndexHostPtr,(size_t ) (n+1) * sizeof(cscColIndex[0]), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(cscVal, cscValHostPtr,(size_t ) (nnz * sizeof(cscVal[0])),cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(thetaT, thetaTHost, (size_t ) (n * f * sizeof(thetaT[0])), cudaMemcpyHostToDevice));
	//CG needs XT
	cudacall(cudaMemcpy(XT, XTHost, (size_t ) (m * f * sizeof(XT[0])), cudaMemcpyHostToDevice));

	cudacall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	//To minimize bank conflics the size of the bank has been set to eight bytes
	//64-bit smem access
	//http://acceleware.com/blog/maximizing-shared-memory-bandwidth-nvidia-kepler-gpus
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	//initialize cublas, cusparse
	cublasHandle_t handle;
	cublascall(cublasCreate(&handle));
	cusparseHandle_t cushandle = 0;
	cusparsecall(cusparseCreate(&cushandle));
	cusparseMatDescr_t descr;
	cusparsecall( cusparseCreateMatDescr(&descr));
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	for(int iter = 0; iter < ITERS ; iter ++){

		//copy csr matrix in
		cudacall(cudaMalloc((void** ) &csrRowIndex,(m + 1) * sizeof(csrRowIndex[0])));
		cudacall(cudaMalloc((void** ) &csrColIndex, nnz * sizeof(csrColIndex[0])));
		cudacall(cudaMalloc((void** ) &csrVal, nnz * sizeof(csrVal[0])));
		cudacall(cudaMemcpy(csrRowIndex, csrRowIndexHostPtr,(size_t ) ((m + 1) * sizeof(csrRowIndex[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(csrColIndex, csrColIndexHostPtr,(size_t ) (nnz * sizeof(csrColIndex[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(csrVal, csrValHostPtr,(size_t ) (nnz * sizeof(csrVal[0])),cudaMemcpyHostToDevice));


		float * ytheta = 0;
		float * ythetaT = 0;
		cudacall(cudaMalloc((void** ) &ytheta, f * m * sizeof(ytheta[0])));
		cudacall(cudaMalloc((void** ) &ythetaT, f * m * sizeof(ythetaT[0])));

		const float alpha = 1.0f;
		const float beta = 0.0f;
		//cusparseScsrmm2 give α ∗ op ( A ) ∗ op ( B ) + β ∗ C where A is a sparce matrix B and C are dense matrices
		cusparsecall (cusparseScsrmm2(cushandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				CUSPARSE_OPERATION_TRANSPOSE, m, f, n, nnz, &alpha, descr, csrVal,
				csrRowIndex, csrColIndex, thetaT, f, &beta, ytheta, m) );

		//printf("*******transpose ytheta use cublas.\n");
		//ytheta: m*f; need ythetaT = (ytheta).T = f*m
		//Summing up X*ThetaR over all iterations and storing in ythetaT
		cublascall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, f, m, &alpha,
				(const float * ) ytheta, m, &beta, ythetaT, f, ythetaT, f));
		//cudaDeviceSynchronize();
		//cudaCheckError();
		cudacall(cudaFree(ytheta));
		cudacall(cudaFree(csrVal));


		int block_dim = f/T10*(f/T10+1)/2;
		   //minimum number of threads is f/2 to load each column of ThetaTs
		if (block_dim < f/2) block_dim = f/2;
		for(int batch_id = 0; batch_id< X_BATCH; batch_id ++){

			int batch_size = 0;
			if(batch_id != X_BATCH - 1)
				batch_size = m/X_BATCH;
			else
				batch_size = m - batch_id*(m/X_BATCH);
			int batch_offset = batch_id * (m/X_BATCH);

			cudacall(cudaMalloc((void** ) &tt, f * f * batch_size * sizeof(float)));

			//updateXByBlock kernel.

				get_hermitianT10<<<batch_size, block_dim, SCAN_BATCH * f/2*sizeof(float2)>>>
					(batch_offset, tt, csrRowIndex, csrColIndex, lambda, m, f, thetaT);
			cudaDeviceSynchronize();
			cudaCheckError();


			float ** devPtrTTHost = 0;
			cudacall(cudaMallocHost( (void** ) &devPtrTTHost, batch_size * sizeof(*devPtrTTHost) ) );
			float **devPtrYthetaTHost = 0;
			cudacall(cudaMallocHost( (void** ) &devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaTHost) ) );
			updateX(batch_size, batch_offset, ythetaT, tt, XT, handle, m, n, f, nnz, devPtrTTHost, devPtrYthetaTHost);
			cudacall(cudaFreeHost(devPtrTTHost));
			cudacall(cudaFreeHost(devPtrYthetaTHost));

			cudacall(cudaFree(tt));
		}

		cudacall(cudaFree(csrRowIndex));
		cudacall(cudaFree(csrColIndex));
		cudacall(cudaFree(ythetaT));

///*
// ________UPDATE THETA_

		float * yTX = 0;
		float * yTXT = 0;
		cudacall(cudaMalloc((void** ) &yTXT, f * n * sizeof(yTXT[0])));
		cudacall(cudaMalloc((void** ) &yTX, n * f * sizeof(yTX[0])));
		cusparsecall( cusparseScsrmm2(cushandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				CUSPARSE_OPERATION_TRANSPOSE, n, f, m, nnz, &alpha, descr, cscVal,
				cscColIndex, cscRowIndex, XT, f, &beta, yTX, n) );
		//cudaDeviceSynchronize();
		//printf("*******transpose yTX \n");
		//yTX: n*f; need yTXT = (yTX).T = f*n
		cublascall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, f, n, &alpha,
				(const float * ) yTX, n, &beta, yTXT, f, yTXT, f));
		cudaDeviceSynchronize();
		cudacall(cudaFree(yTX));

		//in batches, when N is huge
		for(int batch_id = 0; batch_id< THETA_BATCH; batch_id ++){

			int batch_size = 0;
			if(batch_id != THETA_BATCH - 1)
				batch_size = n/THETA_BATCH;
			else
				batch_size = n - batch_id*(n/THETA_BATCH);
			int batch_offset = batch_id * (n/THETA_BATCH);

			float * xx = 0;

			cudacall(cudaMalloc((void** ) &xx, f * f * batch_size * sizeof(xx[0])));
			cudacall( cudaMemset(xx, 0, f*f*batch_size*sizeof(float)) );
			#
			//get_hermitian_theta<<<batch_size, 64>>>(batch_offset, xx, cscRowIndex, cscColIndex, lambda, n);
			//updateThetaByBlock2pRegDsmemTile<<<batch_size, F>>>

				get_hermitianT10<<<batch_size, block_dim, SCAN_BATCH*f*sizeof(float)>>>
					(batch_offset, xx, cscColIndex, cscRowIndex, lambda, n, f, XT);
			cudaDeviceSynchronize();
			cudaCheckError();


			float ** devPtrXXHost = 0;
			cudacall(cudaMallocHost( (void** ) &devPtrXXHost, batch_size * sizeof(*devPtrXXHost) ) );
			float **devPtrYTXTHost = 0;
			cudacall(cudaMallocHost( (void** ) &devPtrYTXTHost, batch_size * sizeof(*devPtrYTXTHost) ) );
			updateTheta(batch_size, batch_offset, xx, yTXT, thetaT, handle, m,  n,  f,  nnz,
					devPtrXXHost, devPtrYTXTHost);


			cudacall(cudaFreeHost(devPtrXXHost));
			cudacall(cudaFreeHost(devPtrYTXTHost));

			cudacall(cudaFree(xx));
		}
		cudacall(cudaFree(yTXT));

		float * errors_train = 0;
		int error_size = 1000;
		cudacall(cudaMalloc((void** ) &errors_train, error_size * sizeof(errors_train[0])));
		cudacall( cudaMemset(errors_train, 0, error_size*sizeof(float)) );

		cudacall(cudaMalloc((void** ) &cooRowIndex, nnz * sizeof(cooRowIndex[0])));
		cudacall(cudaMemcpy(cooRowIndex, cooRowIndexHostPtr,(size_t ) (nnz * sizeof(cooRowIndex[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMalloc((void** ) &csrColIndex, nnz * sizeof(csrColIndex[0])));
		cudacall(cudaMalloc((void** ) &csrVal, nnz * sizeof(csrVal[0])));
		cudacall(cudaMemcpy(csrColIndex, csrColIndexHostPtr,(size_t ) (nnz * sizeof(csrColIndex[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(csrVal, csrValHostPtr,(size_t ) (nnz * sizeof(csrVal[0])),cudaMemcpyHostToDevice));

		RMSE<<<(nnz-1)/256 + 1, 256>>>
				(csrVal, cooRowIndex, csrColIndex, thetaT, XT, errors_train, nnz, error_size, f);
		cudaDeviceSynchronize();
		cudaCheckError();
		cudacall(cudaFree(cooRowIndex));
		cudacall(cudaFree(csrColIndex));
		cudacall(cudaFree(csrVal));

		float* rmse_train = (float*) malloc (sizeof(float));
		cublascall( cublasSasum(handle, error_size, errors_train, 1, rmse_train) );

		cudaDeviceSynchronize();
		printf("--------- Train RMSE in iter %d: %f\n", iter, sqrt((*rmse_train)/nnz));
		cudacall(cudaFree(errors_train));


		float * errors_test = 0;
		cudacall(cudaMalloc((void** ) &errors_test, error_size * sizeof(errors_test[0])));
		cudacall( cudaMemset(errors_test, 0, error_size*sizeof(float)) );

		cudacall(cudaMalloc((void** ) &cooRowIndex_test, nnz_test * sizeof(cooRowIndex_test[0])));
		cudacall(cudaMemcpy(cooRowIndex_test, cooRowIndexTestHostPtr,(size_t ) (nnz_test * sizeof(cooRowIndex_test[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMalloc((void** ) &cooColIndex_test, nnz_test * sizeof(cooColIndex_test[0])));
		cudacall(cudaMalloc((void** ) &cooVal_test, nnz_test * sizeof(cooVal_test[0])));
		cudacall(cudaMemcpy(cooColIndex_test, cooColIndexTestHostPtr,(size_t ) (nnz_test * sizeof(cooColIndex_test[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(cooVal_test, cooValHostTestPtr,(size_t ) (nnz_test * sizeof(cooVal_test[0])),cudaMemcpyHostToDevice));

		RMSE<<<(nnz_test-1)/256, 256>>>(cooVal_test, cooRowIndex_test, cooColIndex_test, thetaT, XT,
				errors_test, nnz_test, error_size, f);
		cudaDeviceSynchronize();
		cudaCheckError();

		cudacall(cudaFree(cooRowIndex_test));
		cudacall(cudaFree(cooColIndex_test));
		cudacall(cudaFree(cooVal_test));

		float* rmse_test = (float*) malloc (sizeof(float));
		cublascall( cublasSasum(handle, error_size, errors_test, 1, rmse_test) );
		cudaDeviceSynchronize();
		final_rmse = sqrt((*rmse_test)/nnz_test);
		printf("--------- Test RMSE in iter %d: %f\n", iter, final_rmse);
		cudacall(cudaFree(errors_test));
//*/
	}
	//copy feature vectors back to host
	cudacall(cudaMemcpy(thetaTHost, thetaT, (size_t ) (n * f * sizeof(thetaT[0])), cudaMemcpyDeviceToHost));
	cudacall(cudaMemcpy(XTHost, XT, (size_t ) (m * f * sizeof(XT[0])), cudaMemcpyDeviceToHost));
	cudacall(cudaFree(thetaT));
	cudacall(cudaFree(XT));
	cudacall(cudaFree(cscVal));
	cudacall(cudaFree(cscColIndex));
	cudacall(cudaFree(cscRowIndex));
	//WARN: do not call cudaDeviceReset inside ALS()
	//because the caller needs to access XT and thetaT which was in the same context
	//cudacall(cudaDeviceReset());
	return final_rmse;
}

int main(int argc, char **argv)
{

  if(argc != 10){
		printf("Usage: give M, N, F, NNZ, NNZ_TEST, lambda, X_BATCH, THETA_BATCH and DATA_DIR.\n");
		printf("E.g., for netflix data set, use: \n");
		printf("./main 17770 480189 100 99072112 1408395 0.048 1 3 ./data/netflix/ \n");
		printf("E.g., for movielens 10M data set, use: \n");
		printf("./main 71567 65133 100 9000048 1000006 0.05 1 1 ./data/ml10M/ \n");
		return 0;
	}

  int f = atoi(argv[3]);
	if(f%T10!=0){
		printf("F has to be a multiple of %d \n", T10);
		return 0;
  }
    int m = atoi(argv[1]);
  	int n = atoi(argv[2]);
  	long nnz = atoi(argv[4]);
  	long nnz_test = atoi(argv[5]);
  	float lambda = atof(argv[6]);
  	int X_BATCH = atoi(argv[7]);
  	int THETA_BATCH = atoi(argv[8]);
  	std::string DATA_DIR(argv[9]);
  	printf("M = %d, N = %d, F = %d, NNZ = %ld, NNZ_TEST = %ld, lambda = %f\nX_BATCH = %d, THETA_BATCH = %d\nDATA_DIR = %s \n",
  			m, n, f, nnz, nnz_test, lambda, X_BATCH, THETA_BATCH, DATA_DIR.c_str());

    int ITERS = 1000;

    int* csrRowIndexHostPtr;
  	cudacall(cudaMallocHost( (void** ) &csrRowIndexHostPtr, (m + 1) * sizeof(csrRowIndexHostPtr[0])) );
  	int* csrColIndexHostPtr;
  	cudacall(cudaMallocHost( (void** ) &csrColIndexHostPtr, nnz * sizeof(csrColIndexHostPtr[0])) );
  	float* csrValHostPtr;
  	cudacall(cudaMallocHost( (void** ) &csrValHostPtr, nnz * sizeof(csrValHostPtr[0])) );
  	float* cscValHostPtr;
  	cudacall(cudaMallocHost( (void** ) &cscValHostPtr, nnz * sizeof(cscValHostPtr[0])) );
  	int* cscRowIndexHostPtr;
  	cudacall(cudaMallocHost( (void** ) &cscRowIndexHostPtr, nnz * sizeof(cscRowIndexHostPtr[0])) );
  	int* cscColIndexHostPtr;
  	cudacall(cudaMallocHost( (void** ) &cscColIndexHostPtr, (n+1) * sizeof(cscColIndexHostPtr[0])) );
  	int* cooRowIndexHostPtr;
  	cudacall(cudaMallocHost( (void** ) &cooRowIndexHostPtr, nnz * sizeof(cooRowIndexHostPtr[0])) );

    //calculate X from thetaT first, need to initialize thetaT
	float* thetaTHost;
	cudacall(cudaMallocHost( (void** ) &thetaTHost, n * f * sizeof(thetaTHost[0])) );

	float* XTHost;
	cudacall(cudaMallocHost( (void** ) &XTHost, m * f * sizeof(XTHost[0])) );

	//initialize thetaT on host
	unsigned int seed = 0;
	srand (seed);
	for (int k = 0; k < n * f; k++)
		thetaTHost[k] = 0.2*((float) rand() / (float)RAND_MAX);

  printf("*******start loading training and testing sets to host.\n");


  int* cooRowIndexTestHostPtr = (int *) malloc(nnz_test * sizeof(cooRowIndexTestHostPtr[0]));

  int* cooColIndexTestHostPtr = (int *) malloc(nnz_test * sizeof(cooColIndexTestHostPtr[0]));

  float* cooValHostTestPtr = (float *) malloc(nnz_test * sizeof(cooValHostTestPtr[0]));


  struct timeval tv0;
  gettimeofday(&tv0, NULL);


  loadCooSparseMatrixBin( (DATA_DIR + "/R_test_coo.data.bin").c_str(), (DATA_DIR + "/R_test_coo.row.bin").c_str(),
  (DATA_DIR + "/R_test_coo.col.bin").c_str(),cooValHostTestPtr, cooRowIndexTestHostPtr, cooColIndexTestHostPtr, nnz_test);

  cout<<"Reached here";
  printf("*******start loading training and testing sets to host.\n");

  loadCSRSparseMatrixBin( (DATA_DIR + "/R_train_csr.data.bin").c_str(), (DATA_DIR + "/R_train_csr.indptr.bin").c_str(),
  (DATA_DIR + "/R_train_csr.indices.bin").c_str(),csrValHostPtr, csrRowIndexHostPtr, csrColIndexHostPtr, m, nnz);

  loadCSCSparseMatrixBin( (DATA_DIR + "/R_train_csc.data.bin").c_str(), (DATA_DIR + "/R_train_csc.indices.bin").c_str(),
  (DATA_DIR +"/R_train_csc.indptr.bin").c_str(), cscValHostPtr, cscRowIndexHostPtr, cscColIndexHostPtr, n, nnz);

  loadCooSparseMatrixRowPtrBin( (DATA_DIR + "/R_train_coo.row.bin").c_str(), cooRowIndexHostPtr, nnz);

  double t0 = seconds();

  cout<<"Reached here";

  doALS(csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr,
			cscRowIndexHostPtr, cscColIndexHostPtr, cscValHostPtr,
			cooRowIndexHostPtr, thetaTHost, XTHost,
			cooRowIndexTestHostPtr, cooColIndexTestHostPtr, cooValHostTestPtr,
			m, n, f, nnz, nnz_test, lambda,
			ITERS, X_BATCH, THETA_BATCH,  0);

	printf("\ndoALS takes seconds: %.3f for F = %d\n", seconds() - t0, f);

  cudaFreeHost(csrRowIndexHostPtr);
	cudaFreeHost(csrColIndexHostPtr);
	cudaFreeHost(csrValHostPtr);
	cudaFreeHost(cscValHostPtr);
	cudaFreeHost(cscRowIndexHostPtr);
	cudaFreeHost(cscColIndexHostPtr);
	cudaFreeHost(cooRowIndexHostPtr);
	cudaFreeHost(XTHost);
	cudaFreeHost(thetaTHost);
	cudacall(cudaDeviceReset());
	printf("\nALS Done.\n");
	return 0;
}
