#include <sys/time.h>
#include <stdio.h>
#include <immintrin.h>

int main(int argc, char const *argv[])
{
	double A[8] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
	double B[8] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
	double C[8];
	__m256d a,b,c;

	a = _mm256_load_pd(&A[0]);
	b = _mm256_load_pd(&B[0]);
	c = _mm256_add_pd(a,b);

	_mm256_store_pd(&C[0],c);

	for(int i=0;i<8;i++)
	{
		printf("C[%d]=%lf\n",i,C[i]);
	}

	return 0;
}