# ============================================================================
# NASA Jet Engine Remaining Useful Life (RUL) Prediction using CNN
# ============================================================================
# This notebook trains and evaluates CNN models for predicting the remaining
# useful life (RUL) of jet engines using the NASA CMaps dataset.
# ============================================================================

import numpy as np
import pandas as pd
import glob
import os
from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.preprocessing import RobustScaler
import time
import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, recall_score

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)
display(HTML("<style>div.output_scroll { height: 35em; }</style>"))

# Configure matplotlib and plotly
reload(plt)
warnings.filterwarnings('ignore')
pio.renderers.default = 'iframe'

# Define custom Plotly template
pio.templates["ck_template"] = go.layout.Template(
    layout_colorway=px.colors.sequential.Viridis,
    layout_autosize=False,
    layout_width=800,
    layout_height=600,
    layout_font=dict(family="Calibri Light"),
    layout_title_font=dict(family="Calibri"),
    layout_hoverlabel_font=dict(family="Calibri Light"),
)
pio.templates.default = 'ck_template+gridon'

# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_train_data(data, factor=0):
    """
    Prepare training data by calculating RUL (Remaining Useful Life).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw training data
    factor : int
        Filter to exclude early cycles (default: 0)
    
    Returns:
    --------
    pd.DataFrame
        Data with RUL column added
    """
    df = data.copy()
    # Calculate max cycles for each engine unit
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL.columns = ['unit_number', 'max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    # RUL = cycles remaining until failure
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'], inplace=True)
    
    return df[df['time_in_cycles'] > factor]


def prepare_test_data(test_data, rul, size):
    """
    Prepare test data by extracting the last N cycles for each unit
    and merging with ground truth RUL values.
    
    Parameters:
    -----------
    test_data : pd.DataFrame
        Raw test data
    rul : pd.DataFrame
        Ground truth RUL values
    size : int
        Number of cycles to extract from the end
    
    Returns:
    --------
    pd.DataFrame
        Test data with RUL values
    """
    df = test_data.copy()
    # Add unit_number to RUL dataframe
    rul["unit_number"] = range(1, len(rul) + 1)
    # Keep last 'size' rows for each device
    df_last = df.groupby("unit_number").tail(size)
    # Merge RUL with test data
    df_merged = df_last.merge(rul, on="unit_number", how="left")
    # Calculate RUL for each cycle in the test sequence
    df_merged['RUL'] = df_merged['RUL'] - (
        df_merged.groupby("unit_number")["time_in_cycles"].transform("max") 
        - df_merged["time_in_cycles"]
    )
    return df_merged


def ready_train(data, size):
    """
    Complete preprocessing pipeline for training data:
    - Drop unused columns
    - Rename columns to meaningful names
    - Calculate RUL values
    - Scale numerical features using RobustScaler
    - Categorize RUL into bins
    - Create sequences
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw training data
    size : int
        Sequence length for creating temporal sequences
    
    Returns:
    --------
    np.ndarray
        Preprocessed sequences ready for model training
    """
    # Drop unused columns
    data.drop(columns=[26, 27], inplace=True)
    
    # Rename columns to meaningful names
    columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 
               'TRA', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf',
               'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 
               'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
    data.columns = columns
    
    # Calculate RUL
    data = prepare_train_data(data)
    data.drop(columns=['NRc'], inplace=True)
    
    # Scale numerical features
    scaler = RobustScaler()
    numerical_cols = ['setting_1', 'T24', 'T30', 'T50', 'P15', 'P30', 'Nf',
                      'Nc', 'Ps30', 'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Categorize RUL into 6 bins: [0-2), [2-5), [5-10), [10-20), [20-50), [50-∞)
    bins = [0, 2, 5, 10, 20, 50, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    data['RUL_cat'] = pd.cut(data['RUL'], bins=bins, labels=labels, include_lowest=True)
    
    # Drop setting columns
    data.drop(columns=['setting_1', 'setting_2'], inplace=True)
    
    # Create temporal sequences
    return data_ready(data, size)


def ready_test(data, rul, size):
    """
    Complete preprocessing pipeline for test data:
    - Drop unused columns
    - Rename columns
    - Prepare test data with RUL values
    - Scale numerical features
    - Categorize RUL into bins
    - Create sequences
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw test data
    rul : pd.DataFrame
        Ground truth RUL values
    size : int
        Sequence length
    
    Returns:
    --------
    np.ndarray
        Preprocessed test sequences
    """
    # Prepare RUL dataframe
    rul.drop(columns=[1], inplace=True)
    rul.columns = ["RUL"]
    
    # Drop unused columns
    data.drop(columns=[26, 27], inplace=True)
    
    # Rename columns
    columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2',
               'TRA', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf',
               'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB',
               'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
    data.columns = columns
    
    # Prepare test data with RUL values
    data = prepare_test_data(data, rul, size)
    data.drop(columns=['NRc'], inplace=True)
    
    # Scale numerical features
    scaler = RobustScaler()
    numerical_cols = ['setting_1', 'T24', 'T30', 'T50', 'P15', 'P30', 'Nf',
                      'Nc', 'Ps30', 'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Categorize RUL
    bins = [-float('inf'), 2, 5, 10, 20, 50, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    data['RUL_cat'] = pd.cut(data['RUL'], bins=bins, labels=labels, include_lowest=True)
    
    # Drop setting columns
    data.drop(columns=['setting_1', 'setting_2'], inplace=True)
    
    # Create temporal sequences
    return data_ready(data, size)


def data_ready(df, seq_size):
    """
    Convert DataFrame into sequences for time-series models.
    Groups data by unit_number and creates sequences of fixed length.
    Pads the last sequence if necessary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed data grouped by unit_number
    seq_size : int
        Length of each sequence
    
    Returns:
    --------
    np.ndarray
        Array of sequences with shape (num_sequences, seq_size, num_features)
    """
    grouped = df.groupby("unit_number")
    sequences = []
    
    # Create sequences for each engine unit
    for name, group in grouped:
        group = group.reset_index(drop=True)
        
        # Split group into chunks of seq_size
        for i in range(0, len(group), seq_size):
            chunk = group.iloc[i:i + seq_size]
            
            # Pad with last row if chunk is shorter than seq_size
            if len(chunk) < seq_size:
                last_row = chunk.iloc[[-1]].copy()
                while len(chunk) < seq_size:
                    chunk = pd.concat([chunk, last_row], ignore_index=True)
            
            sequences.append(chunk)
    
    return np.array(sequences)


# ============================================================================
# DATA LOADING
# ============================================================================

# Sequence length for temporal windows
seq_len = 10

# Load test datasets and corresponding RUL ground truth
print("Loading NASA CMaps dataset...")
test = pd.read_csv("./archive/CMaps/test_FD001.txt", sep=" ", header=None)
rul_test = pd.read_csv("./archive/CMaps/RUL_FD001.txt", sep=" ", header=None)
test_2 = pd.read_csv("./archive/CMaps/test_FD002.txt", sep=" ", header=None)
rul_test_2 = pd.read_csv("./archive/CMaps/RUL_FD002.txt", sep=" ", header=None)
test_3 = pd.read_csv("./archive/CMaps/test_FD003.txt", sep=" ", header=None)
rul_test_3 = pd.read_csv("./archive/CMaps/RUL_FD003.txt", sep=" ", header=None)
test_4 = pd.read_csv("./archive/CMaps/test_FD004.txt", sep=" ", header=None)
rul_test_4 = pd.read_csv("./archive/CMaps/RUL_FD004.txt", sep=" ", header=None)

# Preprocess test datasets
test_data = ready_test(test, rul_test, seq_len)
test_data_2 = ready_test(test_2, rul_test_2, seq_len)
test_data_3 = ready_test(test_3, rul_test_3, seq_len)
test_data_4 = ready_test(test_4, rul_test_4, seq_len)

# Load and preprocess all training datasets
folder_path = "./archive/CMaps/"
train_file_list = glob.glob(os.path.join(folder_path, "train_*.txt"))

train_list = []
for file in train_file_list:
    df = pd.read_csv(file, sep=" ", header=None)
    df_train = ready_train(df, seq_len)
    train_list.append(df_train)

# Concatenate all data
train_data = np.concatenate(train_list, axis=0)
test_data = np.concatenate([test_data, test_data_2, test_data_3, test_data_4], axis=0)

# Feature and label column definitions
feature_cols = ['unit_number', 'T24', 'T30', 'T50', 'P15', 'P30', 'Nf',
                'Nc', 'Ps30', 'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32']
label_cols = ['RUL_cat']

# ============================================================================
# CUSTOM ACTIVATION FUNCTIONS
# ============================================================================

@tf.keras.utils.register_keras_serializable()
def double_relu(x):
    """
    Double ReLU activation (applies ReLU twice with assertion check).
    Used for verification purposes.
    """
    relu_1 = tf.nn.relu(x)
    relu_2 = tf.nn.relu(relu_1)
    
    # Assert that applying ReLU twice gives same result as once
    condition = tf.reduce_all(tf.equal(relu_1, relu_2))
    with tf.control_dependencies([tf.Assert(condition,
                                            data=[tf.constant(
                                                "Error: Relu results are not the same!")])]):
        return relu_1


@tf.keras.utils.register_keras_serializable()
def single_relu(x):
    """Standard ReLU activation."""
    return tf.nn.relu(x)


@tf.function
def taylor_exp_precomputed(x, n_terms=30):
    """
    Compute exponential using Taylor series expansion with precomputed
    factorial coefficients. Optimized for repeated calls.
    
    Parameters:
    -----------
    x : tf.Tensor
        Input values (clipped to [-20, 20] for numerical stability)
    n_terms : int
        Number of Taylor series terms (default: 30)
    
    Returns:
    --------
    tuple
        (result: exponential approximation, helper_array: individual terms)
    """
    # Clip input for numerical stability
    x = tf.clip_by_value(x, -20.0, 20.0)
    
    # Precomputed 1/n! for n=0 to 29
    inv_factorials = tf.constant([
        1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664,
        0.008333333333333333, 0.001388888888888889, 0.0001984126984126984,
        2.48015873015873e-05, 2.7557319223985893e-06, 2.755731922398589e-07,
        2.505210838544172e-08, 2.08767569878681e-09, 1.6059043836821613e-10,
        1.1470745597729724e-11, 7.647163731819816e-13, 4.779477332387385e-14,
        2.8114572543455206e-15, 1.5619206968586225e-16, 8.22063524662433e-18,
        4.110317623312165e-19, 1.9572941063391263e-20, 8.896791392450574e-22,
        3.868170170630684e-23, 1.6117375710961183e-24, 6.446950284384473e-26,
        2.4795962632247973e-27, 9.183689863795545e-29, 3.279174951355552e-30,
        1.1307155004329145e-31
    ], dtype=x.dtype)
    
    result = tf.zeros_like(x)
    x_power = tf.ones_like(x)
    helper_array = []
    
    # Accumulate Taylor series terms
    for n in range(min(n_terms, len(inv_factorials))):
        term = inv_factorials[n] * x_power
        helper_array.append(term)
        result += term
        x_power = x_power * x
    
    return result, helper_array


@tf.keras.utils.register_keras_serializable()
def taylor_sigmoid_precomputed(x, n_terms=30):
    """
    Compute sigmoid using Taylor-approximated exponential.
    Formula: sigmoid(x) = 1 / (1 + exp(-x))
    
    Parameters:
    -----------
    x : tf.Tensor
        Input values
    n_terms : int
        Number of Taylor series terms
    
    Returns:
    --------
    tf.Tensor
        Sigmoid approximation
    """
    # Compute exp(|x|) using Taylor series
    exp_abs_x, helper_array = taylor_exp_precomputed(tf.abs(x), n_terms)
    print(helper_array)
    
    positive_mask = x >= 0
    
    # For x >= 0: sigmoid = exp(x) / (exp(x) + 1)
    pos_result = exp_abs_x / (exp_abs_x + 1.0)
    
    # For x < 0: sigmoid = 1 / (exp(|x|) + 1)
    neg_result = 1.0 / (exp_abs_x + 1.0)
    
    result = tf.where(positive_mask, pos_result, neg_result)
    return result

# ============================================================================
# MODEL CLASS
# ============================================================================

class MyModel:
    """
    CNN + Dense model for RUL classification.
    Supports different activation functions for experimental comparison.
    """
    
    def __init__(self, model_name, num_classes=6, epochs=50, batch_size=20, activation="sigmoid"):
        """
        Initialize model configuration.
        
        Parameters:
        -----------
        model_name : str
            Name for saving the model
        num_classes : int
            Number of RUL categories (default: 6)
        epochs : int
            Training epochs (default: 50)
        batch_size : int
            Batch size for training (default: 20)
        activation : str or callable
            Activation function name or custom function
        """
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.model_name = model_name
        self.history = None
        self.model = None
    
    def build_model(self, input_shape):
        """
        Build CNN + Dense architecture.
        
        Architecture:
        - Conv1D: 64 filters, kernel_size=3
        - MaxPooling1D: pool_size=2
        - Dense: 64 units
        - Flatten
        - Dense: 64 units (×2)
        - Dense: 6 units (output, softmax)
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input (timesteps, features)
        
        Returns:
        --------
        tf.keras.Model
            Compiled model
        """
        seq_model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation=self.activation,
                   input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dense(64, activation=self.activation),
            Flatten(),
            Dense(64, activation=self.activation),
            Dense(64, activation=self.activation),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Wrap in functional API model
        inputs = Input(shape=input_shape, name="input_data")
        outputs = seq_model(inputs)
        model = Model(inputs=inputs, outputs=outputs, name="cnn_dense_model")
        
        # Compile model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        print("✅ Model built successfully.")
        return model
    
    def fit(self, train_data, test_data):
        """
        Prepare data, build model, train, and evaluate.
        
        Parameters:
        -----------
        train_data : np.ndarray
            Training sequences
        test_data : np.ndarray
            Test sequences
        
        Returns:
        --------
        tf.keras.callbacks.History
            Training history
        """
        # Extract features (skip unit_number column) and labels
        X_train = train_data[:, :, 2:]  # Skip first 2 columns
        y_train = train_data[:, -1, 24:]  # Last row, RUL_cat column
        X_test = test_data[:, :, 2:]
        y_test = test_data[:, -1:, 24:]
        
        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train.squeeze(-1), num_classes=self.num_classes)
        y_test = to_categorical(y_test.squeeze(-1), num_classes=self.num_classes)
        
        # Get input shape
        timesteps = X_train.shape[1]
        features = X_train.shape[2]
        input_shape = (timesteps, features)
        print(f"Input shape: {input_shape}")
        print(f"Training samples: {X_train.shape[0]}")
        
        # Build and train model
        model = self.build_model(input_shape)
        
        self.history = model.fit(
            X_train, y_train,
            validation_split=0.3,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        
        print("✅ Training completed successfully.")
        
        # Evaluate on test data
        self.evaluate(X_test, y_test)
        
        # Save model
        os.makedirs("./models/", exist_ok=True)
        self.model.save("./models/" + self.model_name + ".h5")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data and compute metrics.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels (one-hot encoded)
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Convert one-hot to class indices
        y_pred_classes = np.argmax(y_pred, axis=-1)
        y_true_classes = np.argmax(y_test, axis=-1)
        
        # Flatten for metric computation
        y_pred_flat = y_pred_classes.flatten()
        y_true_flat = y_true_classes.flatten()
        
        # Compute metrics
        f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')
        recall = recall_score(y_true_flat, y_pred_flat, average='weighted')
        accuracy = accuracy_score(y_true_flat, y_pred_flat)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall / Sensitivity: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")


# ============================================================================
# TRAINING
# ============================================================================

# Initialize models with different activation functions
models = []
models.append(MyModel(model_name="tanh", num_classes=6, epochs=10,
                      batch_size=20, activation="tanh"))

# Uncomment to experiment with other activations:
models.append(MyModel(model_name="relu", num_classes=6, epochs=10, 
                      batch_size=20, activation="relu"))
models.append(MyModel(model_name="sigmoid", num_classes=6, epochs=10,
                      batch_size=20, activation="sigmoid"))
models.append(MyModel(model_name="double_relu", num_classes=6, epochs=10,
                      batch_size=20, activation=double_relu))
models.append(MyModel(model_name="taylor_sigmoid_precomputed", num_classes=6,
                      epochs=10, batch_size=20, activation=taylor_sigmoid_precomputed))

# Train all models
for model in models:
    model.fit(train_data, test_data)

# ============================================================================
# MODEL EVALUATION & BENCHMARKING
# ============================================================================

def report(folder_path):
    """
    Benchmark trained models: measure inference time and model size.
    
    Parameters:
    -----------
    folder_path : str
        Path containing trained .h5 model files
    """
    files = glob.glob(folder_path + "/*.h5")
    input_shape = (1, 10, 23)
    
    # Fix randomness for reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)
    
    # Create dummy input
    x = tf.convert_to_tensor(np.random.rand(*input_shape).astype(np.float32))
    
    for file in files:
        print("=" * 50)
        print(file)
        
        # Measure model file size
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        
        # Load model
        model = tf.keras.models.load_model(file)
        
        # Wrap inference in tf.function for optimization
        @tf.function
        def inference(x):
            return model(x, training=False)
        
        # Warm-up runs
        for _ in range(500):
            _ = inference(x)
        
        # Synchronize GPU if available
        if tf.config.list_physical_devices('GPU'):
            tf.experimental.sync_devices()
        
        # Benchmark inference time
        num_runs = 1000
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = inference(x)
        if tf.config.list_physical_devices('GPU'):
            tf.experimental.sync_devices()
        end = time.perf_counter()
        
        # Calculate average inference time
        avg_time_ms = (end - start) / num_runs * 1000
        print(f"Average inference time: {avg_time_ms:.3f} ms")


# Run benchmarking report
report("./models/")