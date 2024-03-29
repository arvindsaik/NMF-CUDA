/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 /*
 * define some host utility functions,
 * such as timing and data loading (to host memory)
 */
#ifndef HOST_UTILITIES_H_
#define HOST_UTILITIES_H_
#include <sys/time.h>
#include <fstream>

inline double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}



void loadCSRSparseMatrixBin(const char* dataFile, const char* rowFile, const char* colFile,
		float* data, int* row, int* col, const int m, const long nnz) {
	#ifdef DEBUG
    printf("\n loading CSR...\n");
	#endif
	FILE *dFile = fopen(dataFile,"rb");
	FILE *rFile = fopen(rowFile,"rb");
	FILE *cFile = fopen(colFile,"rb");
	if (!rFile||!dFile||!dFile)
	{
		printf("Unable to open file!");
		return;
	}

	fread(&row[0], 4*(m+1) ,1, rFile);
	fread(&col[0], 4*nnz ,1, cFile);
	fread(&data[0], 4*nnz ,1, dFile);

	fclose(rFile);
	fclose(dFile);
	fclose(cFile);
}

void loadCSCSparseMatrixBin(const char* dataFile, const char* rowFile, const char* colFile,
		float * data, int* row, int* col, const int n, const long nnz) {
	#ifdef DEBUG
    printf("\n loading CSC...\n");
	#endif

	FILE *dFile = fopen(dataFile,"rb");
	FILE *rFile = fopen(rowFile,"rb");
	FILE *cFile = fopen(colFile,"rb");
	if (!rFile||!dFile||!dFile)
	{
		printf("Unable to open file!");
		return;
	}

	fread(&row[0], 4*nnz ,1, rFile);
	fread(&col[0], 4*(n+1) ,1, cFile);
	fread(&data[0], 4*nnz ,1, dFile);

	fclose(rFile);
	fclose(dFile);
	fclose(cFile);
}

void loadCooSparseMatrixRowPtrBin(const char* rowFile, int* row, const long nnz) {
	#ifdef DEBUG
    printf("\n loading COO Row...\n");
	#endif
	FILE *rfile = fopen(rowFile,"rb");
	fread(&row[0], 4*nnz ,1, rfile);
	fclose(rfile);
}

void loadCooSparseMatrixBin(const char* dataFile, const char* rowFile, const char* colFile,
		float* data, int* row, int* col, const long nnz) {
	#ifdef DEBUG
    printf("\n loading COO...\n");
	#endif

	FILE *dFile = fopen(dataFile,"rb");
	FILE *rFile = fopen(rowFile,"rb");
	FILE *cFile = fopen(colFile,"rb");
	if (!rFile||!dFile||!cFile)
	{
		printf("Unable to open file!");
		return;
	}

	fread(&row[0], 4*nnz, 1, rFile);
	fread(&col[0], 4*nnz, 1, cFile);
	fread(&data[0], 4*nnz, 1, dFile);

	fclose(rFile);
	fclose(dFile);
	fclose(cFile);
}



#endif
