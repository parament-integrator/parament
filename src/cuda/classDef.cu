class GPURunner
{
    public:
        GPURunner(); // Constructor
        ~GPURunner(); // Destructor
        void equiprop(cuComplex *carr, float dt, int pts, cuComplex* out);
        void equiprop2(cuComplex *carr, float dt, int pts, cuComplex* out);
        void set_hamiltonian(cuComplex *H0_host, cuComplex *H1_host, int dim_in);
        void readback();
    private:
        // Handles
        cublasHandle_t handle;

        // GPU Arrays and constants
        cuComplex* H0;
        cuComplex* H1;
        cuComplex* one_GPU;
        cuComplex *one_GPU_diag;

        // Dimension of the Hilbert space
        int dim;

        // Currently initialized time steps
        int curr_max_pts;

        // Point arrays
        cuComplex* c0;
        cuComplex* c1;
        cuComplex* X;
        cuComplex* D0;
        cuComplex* D1;

        // Commonly used constants
        cuComplex zero = make_cuComplex(0,0);
        cuComplex one = make_cuComplex(1,0);
        cuComplex two = make_cuComplex(2,0);
        cuComplex mone = make_cuComplex(-1,0);
        cuComplex mtwo = make_cuComplex(-2,0);

        // Helper functions
        void diagonal_add(cuComplex num, cuComplex *C_GPU, int batch_size);
        void expmBatched(cuComplex *M, int pts, cuComplex* tmp_out);

        // check vars
        bool hamiltonian_is_set = false;

        // BESSEL COEFFICIENTS
        cuComplex J[MMAX+1];
        float alpha;
        float beta;

        // Device 
        int numSMs;
};

GPURunner::GPURunner()
{
   
    cublasHandle_t* new_handle;
    new_handle = &handle;
    cublasErrchk(cublasCreate(new_handle));

    gpuErrchk(cudaMalloc(&one_GPU, sizeof(cuComplex)));
    gpuErrchk(cudaMemcpy(one_GPU, &one, sizeof(cuComplex), cudaMemcpyHostToDevice));

    // BESSEL COEFFICIENTS
    alpha = -2.0;
    beta = 2.0;
    J_arr(J, MMAX, 2.0);

    // No points yet allocated
    curr_max_pts = -1;
    cout << curr_max_pts << endl;

    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    
}

GPURunner::~GPURunner()
{    
   
    cublasErrchk(cublasDestroy(handle));

    gpuErrchk(cudaFree(one_GPU));
    
    if (hamiltonian_is_set == true) {
    gpuErrchk(cudaFree(H0));
    gpuErrchk(cudaFree(H1));
    gpuErrchk(cudaFree(one_GPU_diag));
    }

    if (curr_max_pts > 0){
        cudaFree(c0);
        cudaFree(c1);    
        cudaFree(X);    
        cudaFree(D0);
        cudaFree(D1);
        
    }
    
    
    std::cout << "Objected destroyed" << std::endl;
}

void GPURunner::set_hamiltonian(cuComplex *H0_host, cuComplex *H1_host, int dim_in)
{
    dim = dim_in; 

    // Allocate GPU memory
    cudaMalloc(&H0, dim * dim * sizeof(cuComplex));
    cudaMalloc(&H1, dim * dim * sizeof(cuComplex));

    // Transfer to GPU
    cudaMemcpy(H0, H0_host, dim * dim * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(H1, H1_host, dim * dim * sizeof(cuComplex), cudaMemcpyHostToDevice);

    // Helper Arrays
    cudaMalloc(&one_GPU_diag, dim * sizeof(cuComplex));
    cublasCaxpy(handle, dim, &one, one_GPU, 0, one_GPU_diag, 1);

    hamiltonian_is_set = true;
    nvtxMarkA("Set Hamiltonian routine completed");
}

void GPURunner::diagonal_add(cuComplex num, cuComplex *C_GPU, int batch_size)
{
    
    /*return cublasCgemmStridedBatched(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                1, dim, 1,
                num,
                one_GPU, 1, 0,
                one_GPU_diag, 1, 0,
                &one,
                C_GPU, dim+1, dim*dim,
                batch_size);    */

    /*return cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        1, dim, 1,
        num,
        one_GPU, CUDA_C_32F, 1, 0,
        one_GPU_diag, CUDA_C_32F, 1, 0,
        &one,
        C_GPU, CUDA_C_32F, dim+1, dim*dim,
        batch_size, CUDA_C_32F,
        CUBLAS_GEMM_ALGO0);*/
    /*for (int i=0;i<dim;i++){
        cublasCaxpy(handle, batch_size,
            num,
            one_GPU, 0,
            C_GPU+i*dim, dim*dim);
    }*/

    saxpy<<<32*numSMs, 256>>>(num, C_GPU, dim, batch_size);
    return;

}

void GPURunner::readback()
{
    cuComplex* hostprobe = (cuComplex*)malloc(dim * dim * sizeof(cuComplex));
    cudaMemcpy(hostprobe, H0, dim * dim * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    cout << "H0" << endl;
    printcomplex(hostprobe, dim*dim);

    cout << "H1" << endl;
    cudaMemcpy(hostprobe, H1, dim * dim * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    printcomplex(hostprobe, dim*dim);

    free(hostprobe);
}

void GPURunner::equiprop(cuComplex *carr, float dt, int pts, cuComplex *out)
{
    // ///////////////////////////////////////////////////////
    // TRANSFER
    // //////////////////////////////////////////////////////
    // Allocate memory for c arrays if needed
    if (curr_max_pts < pts) {
        if (curr_max_pts > 0){
            cout << "Need to free c arrays" <<endl;
            cudaFree(&c0);
            cudaFree(&c1); 
            cudaFree(&X);
        }
        cout << "Need to malloc c arrays" <<endl;
        gpuErrchk(cudaMalloc(&c0, pts * sizeof(cuComplex)));
        gpuErrchk(cudaMalloc(&c1, pts * sizeof(cuComplex)));
        gpuErrchk(cudaMalloc(&X, dim * dim * pts * sizeof(cuComplex)));
        gpuErrchk(cudaMalloc(&D0, dim * dim * pts * sizeof(cuComplex)));
        gpuErrchk(cudaMalloc(&D1, dim * dim * pts * sizeof(cuComplex)));

        // Memorize how many pts are initalized
        curr_max_pts = pts;
        
        // Fill c0 array with ones
        cublasCscal(handle, pts, &zero, c0, 1);
        cublasErrchk(cublasCaxpy(handle, pts, &one, one_GPU, 0, c0, 1));
    }

    // Transfer c1
    cudaMemcpy(c1, carr, pts * sizeof(cuComplex), cudaMemcpyHostToDevice);

    // ///////////////////////////////////////////////////////
    // EXPAND 
    // ///////////////////////////////////////////////////////
    //cublasCscal(handle,  dim * dim * pts, &zero, X, 1);
    
    cublasErrchk(cublasCgemm(handle,
         CUBLAS_OP_N, CUBLAS_OP_N,
         dim*dim, pts, 1,
         &one,
         H0, dim*dim,
         c0, 1,
         &zero,
         X, dim*dim)); 
    /*cublasCgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim*dim, 1, 1,
        &one,
        H0, dim*dim,0,
        c0, 1,0,
        &zero,
        X, dim*dim,dim*dim,
        pts);*/

    cublasErrchk(cublasCgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim*dim, pts, 1,
        &one,
        H1, dim*dim,
        c1, 1,
        &one,
        X, dim*dim));



            
    // ///////////////////////////////////////////////////////
    // PROPAGATE
    // ///////////////////////////////////////////////////////
    // Rescale dt
    dt = dt*2/(beta-alpha)*2;
    cuComplex dt_complex;
    dt_complex = make_cuComplex(dt,0);

    //printcomplex(&dt_complex,1);
    
    // Loop
    int k = 0;
    //cuComplex ak;

    cuComplex* ptr_accumulate;

    for (k=MMAX; k >= 0; k--) {
        if (k == MMAX){
            cublasErrchk(cublasCscal(handle, pts*dim*dim, &zero, D0, 1));
        } 
        else {
        // D0 = D0 + 2 X @ D1 * dt
        cublasErrchk(cublasCgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim, dim, dim,
            &dt_complex,
            X, dim, dim*dim,
            D1, dim, dim*dim,
            &mone,
            D0, dim, dim*dim,
            pts)) 
        }
        
        // D0 = D0 + I*ak
        diagonal_add(J[k], D0, pts);


        
        // Next step
        k--;
        //cout << k << endl;


        if (k == MMAX-1) {
            ptr_accumulate = &zero;
            //cublasCscal(handle, pts*dim*dim, &zero, D1, 1);
        }         
        if (k == 0){
            ptr_accumulate = &mtwo;
        }

        // D1 = D1 + 2 X @ D0
        cublasErrchk(cublasCgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim, dim, dim,
            &dt_complex,
            X, dim, dim*dim,
            D0, dim, dim*dim,
            ptr_accumulate,
            D1, dim, dim*dim,
            pts)); 
        
                
        // Code to test the arrays

        /*int transferpts = pts*dim*dim;
        cout << "---------- D1";
        cout << " -----------" << endl;
        cuComplex* hostprobe = (cuComplex*)malloc(transferpts * sizeof(cuComplex));
        cudaMemcpy(hostprobe, D1, transferpts* sizeof(cuComplex), cudaMemcpyDeviceToHost);
        printcomplex(hostprobe, transferpts);
        free(hostprobe); */

       // D1 = D1 + I*ak'
       diagonal_add(J[k], D1, pts);

       if (k == MMAX - 1){
           ptr_accumulate = &mone;
       }

    } 
    // D1 contains now the matrix exponentials



    // ///////////////////////////////////////////////////////
    // REDUCE
    // ///////////////////////////////////////////////////////
    // Reduction operation:
    int remain_pts = pts;
    int pad = 0;
    while (remain_pts > 1){

        pad = remain_pts % 2;
        remain_pts = remain_pts/2;

        cublasCgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            dim, dim, dim,
            &one,
            D1          , dim, dim*dim*2,
            D1 + dim*dim, dim, dim*dim*2,
            &zero,
            D1, dim, dim*dim,
            remain_pts);
        if (pad>0){
            // One left over, need to copy to Array
            cublasCcopy(handle, 
                dim*dim,
                D1 + dim*dim*(remain_pts*2), 1,
                D1 + dim*dim*(remain_pts), 1);
            remain_pts += 1;            
        }


    }
    
    // ///////////////////////////////////////////////////////
    // TRANSFER BACK
    // ///////////////////////////////////////////////////////
    cudaMemcpy(out, D1, dim * dim  * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    
    return;
   
}


void GPURunner::equiprop2(cuComplex *carr, float dt, int pts, cuComplex *out)
{
    cuComplex* c0;
    cuComplex* c1;
    cudaMalloc(&c0, pts * sizeof(cuComplex));
    cudaMalloc(&c1, pts * sizeof(cuComplex));

    // Rescale dt
    float beta = 2;
    float alpha = -2;
    dt = dt*2/(beta-alpha);
    cuComplex dt_complex;
    dt_complex = make_cuComplex(dt,0);
    
    // Set c0 
    cublasCscal(handle, pts, &zero, c0, 1);
    cublasCaxpy(handle, pts, &dt_complex, one_GPU, 0, c0, 1);

    // Transfer c1 & scale by dt
    cudaMemcpy(c1, carr, pts * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cublasCsscal(handle, pts, &dt, c1, 1);

    // Initialize iteration arrays
    cuComplex* D0;
    cudaMalloc(&D0, dim * dim * pts * sizeof(cuComplex));
    cuComplex* D1;
    cudaMalloc(&D1, dim * dim * pts * sizeof(cuComplex));

    // Ensure arrays are zeroed
    cublasCscal(handle, pts*dim*dim, &zero, D0, 1);
    cublasCscal(handle, pts*dim*dim, &zero, D1, 1);
    

    // Loop
    int k = 0;
    cuComplex ak;

    //Malloc streams
    // Create a stream for every time step
    cudaStream_t *streams = (cudaStream_t *) malloc(pts*sizeof(cudaStream_t));
    int i;
    for(i=0; i<pts; i++){
        cudaStreamCreate(&streams[i]);
    }

    for (i=0;i<pts;i++){
        // set Cublas stream
        cublasSetStream(handle, streams[i]);
        for (k=MMAX; k >= 0; k--) {
            // D0 = D0 + I*ak
            cublasCgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                1, dim, 1,
                &ak,
                one_GPU, 1,
                one_GPU_diag, 1,
                &one,
                D0+i*dim*dim, dim+1);    
            // D0 = D0 + 2*X0@D1
            // D0 = D0 + c*X1@D1


            k--;
            // D1 = D1 + I*ak'
            // D1 = D1 + 2*X0@D0
            // D1 = D1 + c*X1@D0

        } 
}

 
    for(i=0; i<pts; i++){
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(c0);
    cudaFree(c1);
    cudaFree(D0);
    cudaFree(D1);
    
    return;
}



void destroy_GPURunner(GPURunner *pObj){
    delete pObj;
    pObj = NULL;
}

extern "C"
{
    __declspec(dllexport) GPURunner* GPURunner_new() {return new GPURunner();}
    __declspec(dllexport) void GPURunner_del(GPURunner* cls) {destroy_GPURunner(cls);}
    __declspec(dllexport) void GPURunner_readback(GPURunner* cls) {cls->readback();}
    __declspec(dllexport) void GPURunner_sethamiltonian(GPURunner* cls, cuComplex *H0, cuComplex *H1, int dim){cls->set_hamiltonian(H0, H1, dim);}
    __declspec(dllexport) void GPURunner_equiprop(GPURunner* cls, cuComplex *carr, float dt, int pts, cuComplex* out){cls->equiprop(carr, dt, pts, out);}
}