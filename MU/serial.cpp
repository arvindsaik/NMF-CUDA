#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<iostream>
#include<random>
#include<armadillo>
#include<ctime>
#include<math.h>

using namespace std;
using namespace arma;


void print_mat(mat &A, int m, int n)
{
  cout<<"\nPrinting Matrix:\n";
  for(int i=0;i<m;i++)
  {
    for(int j=0;j<n;j++)
      cout<<A(i, j)<<" ";
    cout<<endl;
  }
}

void randomInit(mat &V)
{


    for (int i = 0; i < V.n_elem; ++i)
	V(i) = rand() / (double)RAND_MAX;
}



int main()
{
  int r, m, n;
  ifstream dataset("dataset.txt");
  float *Ahost;
  //cout << "Enter dimensions of the matrix : ";
  //cin>>m>>n;
  m = 943;
  n = 1682;
  cout << "Enter the rank : ";
  cin>>r;
  // int epochs;
  // cout<<"Enter epochs for training : ";
  // cin>>epochs;
  Ahost = new float[m*n];

  memset(Ahost,0.0,m*n*sizeof(float));

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

  mat V(m, n);
  for (int i=0;i<m;i++)
    for(int j=0;j<n;j++)
      V(i, j) = Ahost[i*n + j];

  size_t columnsToAverage = 5;

  //intialise W and H
  mat W(m, r);
  mat H(r, n);


  W.randu(m, r);
  // W.zeros(m, r);
  //
  //   for (int col = 0; col < r; col++)
  //   {
  //     for (int randCol = 0; randCol < columnsToAverage; randCol++)
  //     {
  //       W.unsafe_col(col) += V.col(rand()%m);
  //     }
  //   }
  //
  //   W /= columnsToAverage;

    // Initialize H to random values.
    H.randu(r, n);
  //randomInit(W);
  //randomInit(H);
  // W.randu(n, r);
  // H.randu(r, n);

  // print_mat(V, m, n);
  // print_mat(W, m, r);
  // print_mat(H, r, n);

  //sleep(5);
  double norm = 0.0, normOld;
  double residue;

  int start_s=clock();
  mat temp(m, n);
  for(int i=0;i<10000;i++)
  {

    W = (W % (V * H.t())) / (W * H * H.t());
    H = (H % (W.t() * V)) / (W.t() * W * H);

    for (size_t j = 0; j < H.n_cols; ++j)
      norm += arma::norm(W * H.col(j), "fro");
    residue = fabs(normOld - norm) / normOld;

    // Store the norm.
    normOld = norm;

    // Increment iteration count
    // iteration++;
    cout << "Iteration " << i << "; residue " << residue << ".\n";

    if(residue<6e-5)
      break;
  }



  temp = W * H;

  //print_mat(temp, m, n);

 double error = 0.0;

  for (int i=0;i<m;i++)
    for(int j=0;j<n;j++)
      error += pow((temp(i, j)-V(i, j)), 2);

  error = sqrt(error);

  cout<<"\nThe total error is :"<<error;

    int stop_s=clock();
    cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC) << endl;

  return 0;
}
