void printcomplex(cuComplex* data, int len) {
    int j = 0;
    for (j = 0; j < len; j++) {
        
        printf("(%5.3f,%5.3fi) ", data[j].x, data[j].y);
    }
    printf("\n");
    return;
}