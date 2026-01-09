#include <stdio.h>
#include <stdlib.h> // For malloc and free
#include <math.h>   // Required for fabs() function

double coeffs[] = {
   1.0, 
        1.0,
        0.5,
        0.16666666666666666,
        0.041666666666666664,
        0.008333333333333333,
        0.001388888888888889,
        0.0001984126984126984,
        2.48015873015873e-05,
        2.7557319223985893e-06,
        2.755731922398589e-07,
        2.505210838544172e-08,
        2.08767569878681e-09,
        1.6059043836821613e-10,
        1.1470745597729724e-11,
        7.647163731819816e-13,
        4.779477332387385e-14,
        2.8114572543455206e-15,
        1.5619206968586225e-16,
        8.22063524662433e-18,
        4.110317623312165e-19,
        1.9572941063391263e-20,
        8.896791392450574e-22,
        3.868170170630684e-23,
        1.6117375710961183e-24,
        6.446950284384473e-26,
        2.4795962632247973e-27,
        9.183689863795545e-29,
        3.279174951355552e-30,
        1.1307155004329145e-31,
};


typedef enum {
    SUCCESS = 0,
    ERROR
} ErrorCode;

double clip_by_value(double x, double min_val, double max_val) {
    if (x < min_val)
        return min_val;
    else if (x > max_val)
        return max_val;
    else
        return x;
}

double* createAndReturnArray(int size) {
    double* arr = (double*)malloc(size * sizeof(double));
    return arr;
}

double* Maclaurin_terms(double x,int num_terms) {
    double* arr = createAndReturnArray(num_terms);
    arr[0] = 1;
    arr[1] = x / coeffs[0];
    for (int i = 2; i < num_terms; i++){
        arr[i] = arr[i-1] * x / (i); 
    }
    return arr;
}
double custom_exp(double *array,int num_terms) {
    double result = 0;
    for (int i = 0;i<num_terms;i++){
        result += array[i];
    }
    return result;
}
double custom_inverse_exp(double *array,int num_terms) {
    double result = 0;
    int sign;
    for (int i = 0;i<num_terms;i++){
        sign = 1 - 2 * (i & 1);
        result += sign * array[i];
    }
    return result;
}

double sigmoid(double input){
    double clipped_value = clip_by_value(input,-10,10);
    int arraySize = 30;
    double* terms = Maclaurin_terms(clipped_value,arraySize);
    double inv_exp = custom_inverse_exp(terms,arraySize);
    free(terms);
    return 1 / (1+inv_exp);
}

double tanh(double input){
    double clipped_value = clip_by_value(input,-10,10);
    int arraySize = 30;
    double* terms = Maclaurin_terms(2*clipped_value,arraySize);
    double exp_value = custom_exp(terms,arraySize);
    free(terms);
    return (exp_value-1)/(exp_value+1);
}

double tanh_error_detection(double input,ErrorCode *error_out){
    double clipped_value = clip_by_value(input,-10,10);
    int arraySize = 30;
    double* terms = Maclaurin_terms(2*clipped_value,arraySize);
    double exp_value = custom_exp(terms,arraySize);
    double y = (exp_value-1)/(exp_value+1);
    //Error detection started
    double exp_inv_value = custom_inverse_exp(terms,arraySize);
    double alpha = (1-y)/(1+y);
    double left = (alpha - 1) / (alpha + 1);
    double right = (exp_inv_value - 1) / (exp_inv_value + 1);
    double epsilon = 0.000001;        // The small tolerance
    if (fabs(left - right) < epsilon) {
        *error_out = SUCCESS;
    } else {
        *error_out = ERROR;
    }

    //Clean up
    free(terms);
    return y;
}



double sigmoid_error_detection(double input,ErrorCode *error_out){
    double clipped_value = clip_by_value(input,-10,10);
    int arraySize = 30;
    double* terms = Maclaurin_terms(clipped_value,arraySize);
    double inv_exp = custom_inverse_exp(terms,arraySize);

    //Error detection started
    double y = 1 / (1+inv_exp);
    double left = y / (1 - y);
    double right = custom_exp(terms,arraySize);
    double epsilon = 0.000001;        // The small tolerance
    if (fabs(left - right) < epsilon) {
        *error_out = SUCCESS;
    } else {
        *error_out = ERROR;
    }

    //Clean up
    free(terms);
    return y;
}
int main(){

    // double res = tanh(11);
    // printf("Result: %f\n", res);
    ErrorCode err;
    double res = tanh_error_detection(1,&err);
    if (err == SUCCESS) {
        printf("Result: %f\n", res);
    } else if (err == ERROR) {
        printf("Fault happened.\n");
    }
}