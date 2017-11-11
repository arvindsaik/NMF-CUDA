#include <bits/stdc++.h>

using namespace std;

int main(){
	ifstream cin("u.data");
	ofstream cout("dataset.txt");
	for(int i=0;i<100000;++i){
		int x;
		cin>>x;
		cout<<x<<"\t";
		cin>>x;
		cout<<x<<"\t";
		cin>>x;
		cout<<x<<endl;
		cin>>x;
	}
	return 0;
}
