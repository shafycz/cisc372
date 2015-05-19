#ifndef PAGERANK_STRAW_H__
#define PAGERANK_STRAW_H__
//// stochastic matrix and page rank functions, initial "strawman" versions ////

void scale(double *C, SparseMatrix S) ;
// Convert S from link matrix to stochastic matrix.
// Also set C to the column sums of the original link matrix.

void strawman_mvpSM(double * y, SparseMatrix S, double *x) ;
// Matrix-vector product, Sparse Matrix:
// For length S->rowdim vector y, length S->coldim vector x, compute y = S*x.

// random surfer clicks a link
void click(SparseMatrix S, double *C, double d, double *y0, double *y1,
		void (* spmv)(double *z, SparseMatrix S, double *y0, int world_rank, int world_size), int world_rank, int world_size) ;
// compute y1 = (1-d)*u + d*S*y0. 
// d is the damping parameter
// y0 is the initial probability distribution vector.
// spmv is a sparse matrix times vector function that computes z = S*y0.
// y1 is the resulting probability distribution vector.


int solve(SparseMatrix S, double *C, double d, double *y0, double *y, double epsilon,
		void (* spmv)(double * w, SparseMatrix S, double * z, int world_rank, int world_size), int world_rank, int world_size) ;
// Repeat click until two successive y's are closer than epsilon.
// d is the damping parameter
// y0 is the initial probability distribution vector.
// Upon return y is the final vector, the page rank vector.
// epsilon is the convergence requirement
// spmv is a sparse matrix times vector function that computes w = S*z.
// The return value is the number of iterations that were used.

/////// implementations ///////////////




void strawman_mvpSM(double * y, SparseMatrix S, double *x) {
// Matrix-vector product, Sparse Matrix:
// For length S->rowdim vector y, length S->coldim vector x, compute y = S*x.
	int i, k;
	for (i = 0; i < S->rowdim; ++i) y[i] = 0; 
	for (k = 0; k < S->nnz; ++k) 
		y[S->row[k]] += S->val[k] * x[S->col[k]];
}

//similar to strawman_mvpSM 
//the work is split up and then at the end it is reduced
void strawman_mpi(double * y, SparseMatrix S, double * x, int world_rank, int world_size) {
	double * local_y;
	local_y = y;

	int block_size = (S->nnz + (world_size - 1)) / world_size;//split up work
	int start_ind = world_rank * block_size;
	int end_ind = start_ind + block_size;

	int j = 0; 
	for (j=0; j<S->rowdim; j++){//loop to set all to 0
		local_y[i] = 0;
	}
	
	int k = start_ind; 
	for (k; k < end_ind && k < S->nnz; k++)//calculates all y's
	{
		local_y[S->row[k]] += S->val[k] * x[S->col[k]];
	}
		
	MPI_Allreduce(local_y, y, S->rowdim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}


void scale(double *C, SparseMatrix S) {
// Convert S from link matrix to stochastic matrix.
// Also set C to the column sums of the original link matrix.
	int i, k;
	for (i = 0; i < S->coldim; ++i) C[i] = 0;
	for (k = 0; k < S->nnz; ++k) C[S->col[k]] += 1;
	for (k = 0; k < S->nnz; ++k) S->val[k] = 1/C[S->col[k]];
	// the zero cols?
}

void click(SparseMatrix S, double *C, double d, double *y0, double *y1,
		void (* spmv)(double *, SparseMatrix, double *, int world_rank, int world_size), int world_rank, int world_size) {
// compute y1 = (1-d)u + d S y0.
	int i;
	spmv(y1, S, y0, world_rank, world_size);
	// true S is given S + 1/n's in cols where given S is entirely zero.
	// get the contribution of the zero columns 
	double xs = 0;
	for (i = 0; i < S->coldim; ++i) if (C[i] == 0) xs += y0[i];
	xs = xs/S->coldim;

	// get contribution of xs and of (1-d)u.
	xs = d*xs;
	double omdbyn = (1-d)/S->coldim; // entry of (1-d)u.
	for (i = 0; i < S->coldim; ++i) y1[i] = d*y1[i] + omdbyn + xs;
}

int solve(SparseMatrix S, double *C, double d, double *y0, double *y, double epsilon,
		void (* spmv)(double *, SparseMatrix, double *, int world_rank, int world_size), int world_rank, int world_size) {
// repeat click until two successive y's are closer than epsilon.
// y0 is the initial probability distribution vector.
// Upon return y is the final vector, the page rank vector.
// The return value is the number of iterations that were used.
	int i, iters = 0;
	double epssq = epsilon*epsilon;
	double disq;
	double *x = (double *) malloc(S->coldim*sizeof(double));
	for (i = 0; i < S->coldim; ++i) x[i] = y0[i];
	do {
		click(S, C, d, x, y, spmv, world_rank, world_size);
		++iters;
		disq = vecdistsq(y, x, S->coldim);
		for (i = 0; i < S->coldim; ++i) x[i] = y[i]; // prep for next iter.
	} while (disq > epssq);
	return iters;
}

#endif // PAGERANK_STRAW_H__
