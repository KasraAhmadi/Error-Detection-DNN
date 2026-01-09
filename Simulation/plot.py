import random, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sympy import factorial, Rational

# ********************************************************************************************************** #

def Random_Flip_bits(value, n):
    int_value = int.from_bytes(np.float64(value).tobytes(), byteorder='little')
    total_bits = Range_injection_floaing

    for _ in range(n):
        bit_to_flip = random.randint(0, total_bits - 1)
        int_value ^= (1 << bit_to_flip)

    flipped_value = np.frombuffer(int_value.to_bytes(8, byteorder='little'), dtype=np.float64)[0]
    return flipped_value

# ********************************************************************************************************** #

def Random_Stuck_one(value, n):
    # Convert the float to its binary representation as a 64-bit integer
    int_value = int.from_bytes(np.float64(value).tobytes(), byteorder='little')
    total_bits = Range_injection_floaing  # We're focusing on the 52 bits related to the mantissa of the float

    for _ in range(n):
        # Randomly choose a bit to force to 1
        bit_to_set = random.randint(0, total_bits - 1)
        int_value |= (1 << bit_to_set)  # Force the chosen bit to 1

    # Convert the 64-bit integer back to a float
    new_value = np.frombuffer(int_value.to_bytes(8, byteorder='little'), dtype=np.float64)[0]
    return new_value

# ********************************************************************************************************** #

def Random_Stuck_zero(value, n):
    # Convert the float to its binary representation as a 64-bit integer
    int_value = int.from_bytes(np.float64(value).tobytes(), byteorder='little')
    total_bits = Range_injection_floaing  # We're focusing on the 52 bits related to the mantissa of the float

    for _ in range(n):
        # Randomly choose a bit to force to 0
        bit_to_clear = random.randint(0, total_bits - 1)
        int_value &= ~(1 << bit_to_clear)  # Force the chosen bit to 0

    # Convert the 64-bit integer back to a float
    new_value = np.frombuffer(int_value.to_bytes(8, byteorder='little'), dtype=np.float64)[0]
    return new_value

# ********************************************************************************************************** #

def Burst_Flipped_bits(value, n):
    int_value = int.from_bytes(np.float64(value).tobytes(), byteorder='little')
    total_bits = Range_injection_floaing

    # Ensure the starting bit leaves room for n consecutive bits
    start_bit = random.randint(0, total_bits - n)

    # Flip the next n consecutive bits
    for i in range(n):
        int_value ^= (1 << (start_bit + i))

    # Convert back to a floating-point number
    flipped_value = np.frombuffer(int_value.to_bytes(8, byteorder='little'), dtype=np.float64)[0]
    return flipped_value

# ********************************************************************************************************** #

def Burst_Stuck_one(value, n):
    int_value = int.from_bytes(np.float64(value).tobytes(), byteorder='little')
    total_bits = Range_injection_floaing

    # Ensure the starting bit leaves room for n consecutive bits
    start_bit = random.randint(0, total_bits - n)

    # Flip the next n consecutive bits
    for i in range(n):
        int_value |= (1 << (start_bit + i))

    # Convert back to a floating-point number
    flipped_value = np.frombuffer(int_value.to_bytes(8, byteorder='little'), dtype=np.float64)[0]
    return flipped_value

# ********************************************************************************************************** #

def Burst_Stuck_zero(value, n):
    int_value = int.from_bytes(np.float64(value).tobytes(), byteorder='little')
    total_bits = Range_injection_floaing

    # Ensure the starting bit leaves room for n consecutive bits
    start_bit = random.randint(0, total_bits - n)

    # Clear the next n consecutive bits
    for i in range(n):
        int_value &= ~(1 << (start_bit + i))

    # Convert back to a floating-point number
    flipped_value = np.frombuffer(int_value.to_bytes(8, byteorder='little'), dtype=np.float64)[0]
    return flipped_value

# ********************************************************************************************************** #

def exp_taylor_no_fault(x, n):
    result = 0
    for k in range(n):
        result += Rational(x)**k / factorial(k)
    return float(result)


def exp_taylor_faulty(x, n):
    result = 0
    helper_data = 0

    fault_indices = random.sample(range(n), term_number)

    for i in range(n):
        reg = Rational(x)**i / factorial(i)
        helper_data = helper_data + ((-1)**i) * reg

        # Faults are injected here on value
        if (Fault_Model == 1): # skipping 
            if i in fault_indices:
                reg = 0

        if (Fault_Model == 2):
            if i in fault_indices:
                reg = Random_Flip_bits(reg, bit_number)

        if (Fault_Model == 3): 
            if i in fault_indices:
                reg = Random_Stuck_one(reg, bit_number)
        
        if (Fault_Model == 4):
            if i in fault_indices:
                reg = Random_Stuck_zero(reg, bit_number)

        if (Fault_Model == 5):
            if i in fault_indices:
                reg = Burst_Flipped_bits(reg, bit_number)

        if (Fault_Model == 6):
            if i in fault_indices:
                reg = Burst_Stuck_one(reg, bit_number)
        
        if (Fault_Model == 7):
            if i in fault_indices:
                reg = Burst_Stuck_zero(reg, bit_number)
        
        if (Fault_Model == 8):
            if i in fault_indices:
                reg = np.random.uniform(-1e10, 1e10)

        result = result + reg
        
    return float(result), float(helper_data)

# ********************************************************************************************************** #

def Sigmoid_exp(x, n, fault_flag):
    if (fault_flag == 0):
        return 1/(1 + exp_taylor_no_fault((-x), n))
    
    elif (fault_flag == 1):
        Result, Helper_data = exp_taylor_faulty((-x), n)
        return 1/(1 + Result), Helper_data

# ********************************************************************************************************** #

def Tanh_exp(x, n, fault_flag):

    if (fault_flag == 0):
        Result = exp_taylor_no_fault(2*x, n)
        return (Result - 1)/(Result + 1)
    
    elif (fault_flag == 1):
        Result, Helper_data = exp_taylor_faulty(2*x, n)
        return (Result - 1)/(Result + 1), (Helper_data - 1)/(Helper_data + 1)
    
# ********************************************************************************************************** #

# --- Wrap your main loop into a function that returns happened_not_detected ---
def run_experiment(threshold, taylor_terms, num_sequences):
    global happened_detected, happened_not_detected, no_effect
    happened_detected = 0
    happened_not_detected = 0
    no_effect = 0

    for i in range(num_sequences):
        v = np.random.uniform(-input_range, input_range)

        # Exponentiation Function
        if(Function == 1):
            y_nf = exp_taylor_no_fault(v, taylor_terms)
            y_f, helper_data = exp_taylor_faulty(v, taylor_terms)

            Left_side = y_f
            Right_side = 1/helper_data
        # ----------------------------------------------------------------------------- #
        
        # Sigmoid Function
        if(Function == 2):
            y_nf = Sigmoid_exp(v, taylor_terms, 0)                  # no faulty output
            y_f, helper_data = Sigmoid_exp(v, taylor_terms, 1)

            Left_side = 1/y_f - 1
            Right_side = 1/helper_data
        # ----------------------------------------------------------------------------- #
        
        # Tanh Function
        if(Function == 3):
            y_nf = Tanh_exp(v, taylor_terms, 0)                       # no faulty output
            y_f, helper_data = Tanh_exp(v, taylor_terms, 1)

            Left_side = y_f
            Right_side = (-1) * helper_data
        # ----------------------------------------------------------------------------- #

        if (y_nf == y_f):
            no_effect += 1
        if (abs(Left_side - Right_side)) < threshold:
            happened_not_detected += 1
        if (abs(Left_side - Right_side)) > threshold:
            happened_detected += 1

    return happened_not_detected


Function = 3
Fault_Model = 0 
term_number = 0
bit_number = 0

# ********************************************************************************************************** #
# General parameters
Range_injection_floaing = 52 # 64 bits floating point based on IEEE double precision format
input_range = 3
num_sequences = 1000

# --- Parameter sweep ---
threshold_values = np.logspace(-20, -5, 10)   # 10 thresholds from 1e-16 to 1e-8
taylor_terms_values = np.arange(10, 60, 5)   # 10, 15, 20, …, 55

X, Y = np.meshgrid(threshold_values, taylor_terms_values)
Z = np.zeros_like(X, dtype=float)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = run_experiment(X[i, j], Y[i, j], num_sequences)

# --- 3D Surface Plot ---
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(np.log10(X), Y, Z, cmap='viridis', edgecolor='k', alpha=0.5)

functions_texts = {
    1: "'EXPONENTIATION'",
    2: "'SIGMOID'",
    3: "'TANH'",
    }
text_function = functions_texts.get(Function, "'UNKNOWN FUNCTION'")


ax.set_xlabel('log10(threshold)')
ax.set_ylabel('Taylor Terms')
ax.set_zlabel('Result')
ax.set_title(text_function)


plt.show()

