#include <Arduino.h>
#define SAMPLES 500
unsigned long startTime;

unsigned long sigmoid_elapsedTime;
unsigned long error_sigmoid_elapsedTime;

unsigned long tanh_elapsedTime;
unsigned long error_tanh_elapsedTime;

// Coefficients for Maclaurin series
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

double maclaurin_array[30];

void Maclaurin_terms(double x, int num_terms) {
    maclaurin_array[0] = 1;
    maclaurin_array[1] = x / coeffs[0];
    for (int i = 2; i < num_terms; i++){
        maclaurin_array[i] = maclaurin_array[i-1] * x / (i); 
    }
}

double custom_exp(int num_terms) {
    double result = 0;
    for (int i = 0; i < num_terms; i++){
        result += maclaurin_array[i];
    }
    return result;
}

double custom_inverse_exp(int num_terms) {
    double result = 0;
    int sign;
    for (int i = 0; i < num_terms; i++){
        sign = 1 - 2 * (i & 1);
        result += sign * maclaurin_array[i];
    }
    return result;
}

double sigmoid(double input){
    double clipped_value = clip_by_value(input, -10, 10);
    int arraySize = 30;
    Maclaurin_terms(clipped_value, arraySize);
    double inv_exp = custom_inverse_exp(arraySize);
    return 1 / (1 + inv_exp);
}

double tanh_func(double input){
    double clipped_value = clip_by_value(input, -10, 10);
    int arraySize = 30;
    Maclaurin_terms(2 * clipped_value, arraySize);
    double exp_value = custom_exp(arraySize);
    return (exp_value - 1) / (exp_value + 1);
}

double tanh_error_detection(double input, ErrorCode *error_out){
    double clipped_value = clip_by_value(input, -10, 10);
    int arraySize = 30;
    Maclaurin_terms(2 * clipped_value, arraySize);
    double exp_value = custom_exp(arraySize);
    double y = (exp_value - 1) / (exp_value + 1);
    
    // Error detection started
    double exp_inv_value = custom_inverse_exp(arraySize);
    double alpha = (1 - y) / (1 + y);
    double left = (alpha - 1) / (alpha + 1);
    double right = (exp_inv_value - 1) / (exp_inv_value + 1);
    double epsilon = 0.000001;
    if (fabs(left - right) < epsilon) {
        *error_out = SUCCESS;
        return y;

    } else {
        *error_out = ERROR;
        return 0;
    }

}

double sigmoid_error_detection(double input, ErrorCode *error_out){
    double clipped_value = clip_by_value(input, -10, 10);
    int arraySize = 30;
    Maclaurin_terms(clipped_value, arraySize);
    double inv_exp = custom_inverse_exp(arraySize);

    // Error detection started
    double y = 1 / (1 + inv_exp);
    double left = y / (1 - y);
    double right = custom_exp(arraySize);
    double epsilon = 0.000001;
    
    if (fabs(left - right) < epsilon) {
        *error_out = SUCCESS;
        return y;

    } else {
        *error_out = ERROR;
        return 0;
    }

}

// Function to measure clock cycles
unsigned long measureClockCycles(void (*func)(void)) {
    unsigned long start = micros();
    func();
    unsigned long end = micros();
    return (end - start) * (F_CPU / 1000000UL); // Convert microseconds to clock cycles
}

// Store test inputs in PROGMEM (flash memory)
// Pre-generated random values between -10 and 10 (1000 inputs)
const double test_inputs[SAMPLES] PROGMEM = {
  3.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  9.12, -3.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  6.54, -4.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  4.67, -8.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  8.12, -2.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  2.78, -5.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  1.78, -7.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  7.34, -1.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  5.78, -3.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  6.12, -9.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
 3.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  9.12, -3.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  6.54, -4.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  4.67, -8.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  8.12, -2.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  2.78, -5.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  1.78, -7.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  7.34, -1.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  5.78, -3.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  6.12, -9.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
    3.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  9.12, -3.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  6.54, -4.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  4.67, -8.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  8.12, -2.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  2.78, -5.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  1.78, -7.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  7.34, -1.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  5.78, -3.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  6.12, -9.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
      0.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  0.12, -0.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  6.54, -0.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  4.67, -0.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  8.12, -0.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  0.78, -0.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  1.78, -0.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  7.34, -0.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  5.78, -0.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  6.12, -0.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
      3.45, -7.82, 1.23, -9.54, 5.67, -2.34, 8.91, -4.56, 0.78, -6.89,
  0.12, -0.45, 2.56, -8.76, 4.32, -1.23, 7.89, -5.67, 3.21, -9.87,
  0.54, -0.32, 1.09, -7.65, 8.34, -2.10, 5.43, -6.78, 9.00, -3.56,
  0.67, -0.23, 2.34, -5.89, 7.56, -1.45, 6.78, -9.34, 3.90, -4.67,
  0.12, -0.56, 1.34, -7.23, 5.89, -3.78, 9.45, -6.12, 4.56, -8.90,
  0.78, -0.34, 7.01, -1.56, 6.23, -9.67, 3.67, -4.12, 8.56, -2.89,
  0.78, -0.90, 5.12, -3.23, 9.34, -6.45, 4.23, -8.67, 2.90, -5.78,
  0.34, -0.34, 6.56, -9.23, 3.12, -4.89, 8.67, -2.34, 1.45, -7.01,
  0.78, -0.89, 9.56, -6.34, 4.01, -8.12, 2.67, -5.67, 7.45, -1.23,
  0.12, -0.34, 3.89, -4.45, 8.34, -2.78, 1.23, -7.45, 5.45, -3.12,
};


// Read test input from PROGMEM
double getTestInput(int index) {
    return pgm_read_float(&test_inputs[index]);
}

// Global index for current test input
int input_index = 0;

// Wrapper functions for cycle measurement
void sigmoid_wrapper() {
    sigmoid(getTestInput(input_index));
}

void tanh_wrapper() {
    tanh_func(getTestInput(input_index));
}

void tanh_error_detection_wrapper() {
    // tanh_func(getTestInput(input_index));
    // tanh_func(getTestInput(input_index));
    ErrorCode err;
    tanh_error_detection(getTestInput(input_index), &err);
}

void sigmoid_error_detection_wrapper() {
    ErrorCode err;
    sigmoid_error_detection(getTestInput(input_index), &err);
}

void setup() {
    Serial.begin(9600);
    delay(2000); // Wait for serial to initialize
    
    Serial.println("========================================");
    Serial.println("Arduino UNO Clock Cycle Profiler");
    Serial.println("========================================");
    Serial.print("F_CPU: ");
    Serial.print(F_CPU);
    Serial.println(" Hz");
    Serial.println("Test inputs stored in PROGMEM (flash)");
    Serial.println("Number of test inputs: 1000");
    Serial.println();
}

void loop() {

    //sigmoid
    startTime = micros();
    for (input_index = 0; input_index < SAMPLES; input_index++) {
        sigmoid(getTestInput(input_index));
    }
    sigmoid_elapsedTime = micros() - startTime;
    Serial.print("sigmoid Execution Time: ");
    Serial.print(sigmoid_elapsedTime);
    Serial.println(" microseconds");

    // //sigmoid error
    // startTime = micros();
    // for (input_index = 0; input_index < SAMPLES; input_index++) {
    //     ErrorCode err;
    //     sigmoid_error_detection(getTestInput(input_index), &err);
    // }
    // error_sigmoid_elapsedTime = micros() - startTime;
    // Serial.print("sigmoid_error_detection Execution Time: ");
    // Serial.print(error_sigmoid_elapsedTime);
    // Serial.println(" microseconds");

    // //tanh
    // startTime = micros();
    // for (input_index = 0; input_index < SAMPLES; input_index++) {
    //     tanh_func(getTestInput(input_index));
    // }
    // tanh_elapsedTime = micros() - startTime;
    // Serial.print("tanh Execution Time: ");
    // Serial.print(tanh_elapsedTime);
    // Serial.println(" microseconds");

    //tanh error
    // startTime = micros();
    // for (input_index = 0; input_index < SAMPLES; input_index++) {
    //     ErrorCode err;
    //     tanh_error_detection(getTestInput(input_index), &err);
    // }
    // error_tanh_elapsedTime = micros() - startTime;
    // Serial.print("tanh_error_detection Execution Time: ");
    // Serial.print(error_tanh_elapsedTime);
    // Serial.println(" microseconds");
    Serial.println("========== END ==========");

    // Serial.print("Sigmoid overhead:");
    // double x = (error_sigmoid_elapsedTime-sigmoid_elapsedTime)/sigmoid_elapsedTime;
    // Serial.println(x,7);
    
    delay(5000); // Wait 5 seconds before next measurement
}