import random, numpy as np
from sympy import factorial, Rational, log, floor



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
                scale = floor(log(abs(reg), 10))
                reg = np.random.randn() * (10**scale)

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

def main():

    happened_detected = 0       # fault changed the output and it got detected
    happened_not_detected = 0   # fault changed the output and it remained undetected
    no_effect = 0               # either fault injected and had no impact to the output, or it did not inject at all
    
    random.seed()

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

        # "Threshold" version or "Exact" version? for the first one? 

        if (abs(y_nf - y_f)) < threshold_output:                   # This is the "Threshold" version which is 100% always
        #if (y_nf == y_f):                                         # This is the "Exact" version. 
            no_effect += 1                                         # Faults have not changed the output
        
        elif (abs(Left_side - Right_side)) < threshold_countermeasure:        
        # if (abs(Left_side - Right_side)) < threshold_countermeasure:        # output has changed yet our countermeasure is not detecting a fault (equation holds!)
            happened_not_detected += 1
        
        elif (abs(Left_side - Right_side)) > threshold_countermeasure:
        # if (abs(Left_side - Right_side)) > threshold_countermeasure:        # output has changed and our countermeasure is detecting it (equation does not hold)
            happened_detected += 1

    detection_rate = (1 - (happened_not_detected/num_sequences))*100

# ********************************************************************************************************** #
    # For Printing
    fault_models_texts = {
        0: "'NO FAULT'",
        1: "'SKIPPING'",
        2: "'RANDOM BIT FLIPPING'",
        3: "'RANDOM STUCK @ 1'",
        4: "'RANDOM STUCK @ 0'",
        5: "'BURST BIT FLIPPING'",
        6: "'BURST STUCK @ 1'",
        7: "'BURST STUCK @ 0'",
        8: "'ALTER'"
        }
    text_model = fault_models_texts.get(Fault_Model, "'UNKNOWN FAULT'")

    functions_texts = {
        1: "'Exponentiation'",
        2: "'sigmoid'",
        3: "'tanh'",
        }
    text_function = functions_texts.get(Function, "'UNKNOWN FUNCTION'")

    print("Targeted Function:", text_function)
    print("Injection Model:", text_model)
    print("Number of Faulty terms:", term_number)
    print("Number of Faulty bits in each faulty term:", bit_number)
    print("Detection rate:", detection_rate)
    print("-----------------------------------")

# ********************************************************************************************************** #
    # for functionality tests replace elif s with if and uncomment these.
    # Basically if there are no faults, happened_not_detected must be 1000. The ratio here can be adjusted with taylor terms and threshold
    # After fixing a good value then we change the ifs back to elifs.
    
    # print("The faults had no effect on the output", no_effect)
    # print("Fault happened but our countermeasure could not detect it", happened_not_detected)
    # print("Fault happened and our countermeasure has detected it", happened_detected)
    # print("----------------------------------------------------------------------")

# ********************************************************************************************************** #
# Simulation parameters
    # Function
        # 1 Exponentiation
        # 2 Sigmoid
        # 3 Tanh
    # Fault_Model
        # 1 => Skip "term_number" terms
        # 2 => random flip "bit_number" bits
        # 3 => random stuck at 1 "bit_number" bits
        # 4 => random stuck at 0 "bit_number" bits
        # 5 => Burst flip "bit_number" bits
        # 6 => Burst stuck at 1 "bit_number" bits
        # 7 => Burst stuck at 0 "bit_number" bits
        # 8 => Alter completely "term_number" terms
    # term_number: Number of faulty terms
    # bit_number: Number of faulty bits in each faulty term

# ********************************************************************************************************** #
# Fault injection parameters
Function = 1
Fault_Model = 2 
term_number = 3
bit_number = 5

# ********************************************************************************************************** #
# General parameters
Range_injection_floaing = 52 # 64 bits floating point based on IEEE double precision format
num_sequences = 1000

# when even fault is not happening it is possible that the countermeasure detects a fault.
# This is also called round-off error- Change the elif to if to see if it works without faults.
# this depends on input range, threshold and taylor terms and the function it self.
# we selected the following parameter settings based on optimization in 3d plotting
    # (input range, taylor terms, Threshold2)
        # Expo =>    (3, 30, 10^-13)
        # sigmoid => (3, 30, 10^-13)
        # tanh =>    (3, 40, 10^-14)

input_range = 3
taylor_terms = 40
threshold_countermeasure = 1e-14          

threshold_output = 1e-15          # This is for checking if faults have changed the ourput or not
                                  # 64 is very close to EXACT Version. 
                                  # if we make it loose, for example equal to threshold2, it gives us better detection ratios

# Manual
# main()

for Function in [3]:
    for Fault_Model in [5, 6, 7]:
        for term_number in [1, 3, 6]:
            for bit_number in [2, 5]:
                main()




# # Automated
# for Function in [1, 2, 3]:
#     for Fault_Model in [1, 2, 3, 4, 5, 6, 7, 8]:
#         for term_number in [1, 4, 8, 15]:
#             for bit_number in [1, 4, 10, 20]:
#                 main()