#include <stdio.h>
#include "pagerank-util.h"
#include <mpi.h>

// pagerank-straw provides straw_mvpSM(), scale(), solve().
// some subset of these functions (definitely including straw_mvpSM) will be modified for performance.
#include "pagerank-straw.h"

/////////////////////////////////////////////////////////////
// The imagined pagerank-serial.h contains my (student) serial_mvpSM and possibly other reworked functions.
//#include "pagerank-serial.h"
// ...or I put it right here:
void serial_mvpSM(double * w, SparseMatrix S, double * z) {
	// this is a stub.  ..replace this with an improved serial matrix vector product.
	strawman_mvpSM(w, S, z);
}
/////////////////////////////////////////////////////////////

void uniform(double *x, int n) {
// Vector x has length n, is already allocated.  All entries are set to 1/n.
	int i;
	double ninv = 1.0/n;
	for (i = 0; i < n; ++i) x[i] = ninv;
}

int main(int argc, char* argv[]){
	// hello
	int n = 100;
	double d = 0.85; 
	double eps = 0.000000001;
	if (argc <= 1 || argc > 4) {
		printf("usage: %s num-pages damping-factor(0.85) epsilon(0.0001)\n", argv[0]);
		return 0;
	}
	if (argc > 1) n = atoi(argv[1]);
	if (argc > 2) d = atof(argv[2]);
	if (argc > 3) eps = atof(argv[3]);

	// build vectors
	double *y0  = (double *) malloc(n*sizeof(double)); // holds initial page probabilities
	double *y  = (double *) malloc(n*sizeof(double)); // holds intermediate and ultimate probs.
	uniform(y0, n);
	
	MPI_Init(0,0);
	
	
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);//get number of processes
	
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);//rank of the process
	
	if (world_rank==0) {	
		printf("initial probabilities\n");
		printvec(y0, n);
	}
	
	// build matrix
	struct SparseMatrixHandle SH;
	SparseMatrix S = &SH;
	randomLM(S, n); // The link matrix
	if (n <= 100) writeSM(S, 50);
	if (world_rank==0) {
		printf("dimension is %d, nnz is %d, damper is %f, epsilon is %g\n\n", S->coldim, S->nnz, d, eps);
	}
	double *C = (double *) malloc(S->coldim*sizeof(double));
	scale(C, S); // convert link matrix to stochastic matrix (col sums are 1).

	double start, elapsed, local_elapsed;
	int iters;

	start = MPI_Wtime();
	iters = solve(S, C, d, y0, y, eps, strawman_mpi, world_rank, world_size);

	local_elapsed = MPI_Wtime() - start;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (world_rank==0) {	
		printf("%f secs per iter, %d iters to return final page rank prob\n", iters, elapsed);
		printvec(y, n); 
		printf("\n");
	}

	// good bye
	free(C); free(y); free(y0); free(SH.row); free(SH.col); free(SH.val);
	return 0;
}
