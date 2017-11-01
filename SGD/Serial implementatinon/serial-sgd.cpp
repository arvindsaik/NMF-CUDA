#include<bits/stdc++.h>

#include <armadillo>
using namespace std;

using namespace arma;

int main(){
  cout<<"enter the dimensions of matrix : ";
  int m,n;
  cin>>m>>n;

  mat A = randu<mat>(m,n);
  A = A*99; //random

  // A.print("A : ");

  int epochs;
  cout<<"Enter epochs for training : \n";
  cin>>epochs;
  int k;
  cout<<"Enter k for output : \n";
  cin>>k;
  double alpha;
  cout<<"Enter alpha for training : \n";
  cin>>alpha;
  double lamda;
  cout<<"Enter lamda (regularisation variable) for training : \n";
  cin>>lamda;
  mat B = randu<mat>(m,k); //random values from 0 to 1
  mat C = randu<mat>(k,n);


  for(int i=0;i<epochs;++i){
    for(int u=0;u<m;++u){
      for(int v=0;v<n;++v){
        mat errorMat  = (A(u,v) - B.row(u)*C.col(v));
        double error = errorMat(0,0);
        B.row(u) = B.row(u) + alpha*(error*(C.col(v).t()) - lamda*(B.row(u)));
        C.col(v) = C.col(v) + alpha*(error*(B.row(u).t()) - lamda*(C.col(v)));
      }
    }
  }
  double sumError = 0;
  for(int u=0;u<m;++u){
    for(int v=0;v<n;++v){
      mat errorMat  = (A(u,v) - B.row(u)*C.col(v));
      sumError += errorMat(0,0)*errorMat(0,0);
    }
  }
  sumError = sqrt(sumError);

  // B.print("B : ");
  // C.print("C : ");
  // (B*C).print("Product : ");
  cout << "RMS error : " << sqrt(sumError) << endl;
}
