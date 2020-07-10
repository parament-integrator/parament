cuComplex imag_power(int k) {
    if      (k%4 == 0){
        return make_cuComplex(1, 0);
    }
    else if (k%4 ==1 ){
        return make_cuComplex(0, -1);
    }
    else if (k%4 == 2){
        return make_cuComplex(-1, 0);
    }
    else {
        return make_cuComplex(0, 1);
    }
}

void J_arr(cuComplex* arr, int mmax, double c) {
    int i = 0;
    for (i = 0; i < mmax + 1; i++) {
        arr[i] = cuCmulf(imag_power(i),make_cuComplex(_jn(i,c), 0));
    }
    return;
}
