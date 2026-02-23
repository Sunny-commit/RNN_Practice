# 🚀 RNN Practice - Recurrent Neural Networks

A comprehensive **deep learning guide for Recurrent Neural Networks** covering LSTM, GRU, and sequence modeling for time-series prediction, NLP tasks, and sequential data analysis with practical implementations and applications.

## 🎯 Overview

This project covers:
- ✅ LSTM fundamentals & architecture
- ✅ GRU (Gated Recurrent Units)
- ✅ Bidirectional RNNs
- ✅ Attention mechanisms
- ✅ Sequence-to-Sequence models
- ✅ Real-world applications

## 🧠 RNN Fundamentals

### Basic RNN Cell

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SimpleRNNCell:
    """Understand RNN cell mechanics"""
    
    def __init__(self, units, input_size):
        """
        units: Hidden state dimension
        input_size: Input feature dimension
        """
        self.units = units
        self.input_size = input_size
        
        # Weights
        self.W_x = np.random.randn(input_size, units) * 0.001  # Input weights
        self.W_h = np.random.randn(units, units) * 0.001       # Recurrent weights
        self.b = np.zeros((1, units))                           # Bias
    
    def forward(self, x, h_prev):
        """
        Forward pass
        x: Input (batch_size, input_size)
        h_prev: Previous hidden state (batch_size, units)
        """
        self.x = x
        self.h_prev = h_prev
        
        # Compute hidden state
        self.h = np.tanh(np.dot(x, self.W_x) + np.dot(h_prev, self.W_h) + self.b)
        
        return self.h
    
    def backward(self, dh, learning_rate=0.01):
        """Backpropagation through time"""
        # Derivative of tanh
        dh_raw = dh * (1 - self.h ** 2)
        
        # Gradients
        dW_x = np.dot(self.x.T, dh_raw)
        dW_h = np.dot(self.h_prev.T, dh_raw)
        db = np.sum(dh_raw, axis=0, keepdims=True)
        
        # Update weights
        self.W_x -= learning_rate * dW_x
        self.W_h -= learning_rate * dW_h
        self.b -= learning_rate * db
        
        # Gradient for previous layer
        dh_prev = np.dot(dh_raw, self.W_h.T)
        
        return dh_prev

# Example
rnn_cell = SimpleRNNCell(units=64, input_size=32)
h_prev = np.zeros((batch_size, 64))
x_t = np.random.randn(batch_size, 32)
h_next = rnn_cell.forward(x_t, h_prev)
```

### LSTM Architecture

```python
class LSTMCell:
    """Long Short-Term Memory cell"""
    
    def __init__(self, units):
        self.units = units
        self.W = None
        self.b = None
    
    def build(self, input_size):
        """Initialize weights"""
        # Concatenate input + hidden -> 4*units (for 4 gates)
        self.W = np.random.randn(input_size + self.units, 4 * self.units) * 0.001
        self.b = np.zeros((1, 4 * self.units))
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass
        x: Input (batch_size, input_size)
        h_prev: Previous hidden state
        c_prev: Previous cell state
        """
        # Concatenate input and previous hidden
        x_combined = np.concatenate([x, h_prev], axis=1)
        
        # Compute gate outputs
        z = np.dot(x_combined, self.W) + self.b
        
        # Split into 4 gates
        size = self.units
        z_i = z[:, :size]      # Input gate
        z_f = z[:, size:2*size]     # Forget gate
        z_c = z[:, 2*size:3*size]   # Cell gate
        z_o = z[:, 3*size:]    # Output gate
        
        # Apply activations
        i = sigmoid(z_i)       # Input gate
        f = sigmoid(z_f)       # Forget gate
        c_tilde = np.tanh(z_c) # Cell candidate
        o = sigmoid(z_o)       # Output gate
        
        # Update cell state
        c = f * c_prev + i * c_tilde
        
        # Update hidden state
        h = o * np.tanh(c)
        
        self.cache = (x, h_prev, c_prev, i, f, c_tilde, o, c, z)
        
        return h, c
    
    def backward(self, dh, dc, learning_rate=0.01):
        """Backpropagation through LSTM"""
        # Extract cache
        x, h_prev, c_prev, i, f, c_tilde, o, c, z = self.cache
        
        # Gradients
        dt = dh * o * (1 - np.tanh(c) ** 2) + dc
        dc_prev = dt * f
        
        df = dt * c_prev
        di = dt * c_tilde
        dc_tilde = dt * i
        do = dh * np.tanh(c)
        
        # Gate gradients
        dz_i = di * i * (1 - i)
        dz_f = df * f * (1 - f)
        dz_c = dc_tilde * (1 - c_tilde ** 2)
        dz_o = do * o * (1 - o)
        
        dz = np.concatenate([dz_i, dz_f, dz_c, dz_o], axis=1)
        
        dW = np.dot(np.concatenate([x, h_prev], axis=1).T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        
        # Update
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

### GRU Architecture

```python
class GRUCell:
    """Gated Recurrent Unit - simpler than LSTM"""
    
    def __init__(self, units):
        self.units = units
        self.W = None
        self.b = None
    
    def forward(self, x, h_prev):
        """
        GRU forward pass (2 gates instead of LSTM's 3)
        """
        x_combined = np.concatenate([x, h_prev], axis=1)
        z = np.dot(x_combined, self.W) + self.b
        
        # Reset and update gates
        r = sigmoid(z[:, :self.units])      # Reset gate
        u = sigmoid(z[:, self.units:2*self.units])  # Update gate
        
        # Candidate hidden state
        h_candidate = np.tanh(
            z[:, 2*self.units:] + 
            r * np.dot(h_prev, self.W[:self.units, 2*self.units:])
        )
        
        # New hidden state
        h = (1 - u) * h_candidate + u * h_prev
        
        return h
```

## 📊 Time-Series Forecasting with RNN

```python
class TimeSeriesRNN:
    """RNN for time-series prediction"""
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.model = self._build_model()
    
    def _build_model(self):
        """Build LSTM model for time series"""
        model = keras.Sequential([
            layers.LSTM(
                64,
                activation='relu',
                input_shape=(self.sequence_length, 1),
                return_sequences=True
            ),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # Single output
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_data(self, data, train_size=0.8):
        """Prepare sequences for training"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        split = int(len(X) * train_size)
        
        return {
            'X_train': X[:split],
            'y_train': y[:split],
            'X_val': X[split:],
            'y_val': y[split:]
        }
    
    def train(self, data, epochs=50):
        """Train model"""
        dataset = self.prepare_data(data)
        
        history = self.model.fit(
            dataset['X_train'],
            dataset['y_train'],
            validation_data=(dataset['X_val'], dataset['y_val']),
            epochs=epochs,
            batch_size=32
        )
        
        return history
    
    def forecast(self, data, steps=10):
        """Forecast future values"""
        sequence = data[-self.sequence_length:].reshape(1, -1, 1)
        forecasts = []
        
        for _ in range(steps):
            pred = self.model.predict(sequence, verbose=0)[0, 0]
            forecasts.append(pred)
            
            # Update sequence
            sequence = np.append(sequence[0, 1:, 0], pred)
            sequence = sequence.reshape(1, -1, 1)
        
        return np.array(forecasts)
```

## 🔤 NLP with RNNs

### Text Classification

```python
class TextClassificationRNN:
    """LSTM for text classification"""
    
    def __init__(self, vocab_size=10000, max_length=100, num_classes=2):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build text classification model"""
        model = keras.Sequential([
            layers.Embedding(self.vocab_size, 128, input_length=self.max_length),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def encode_text(self, texts, tokenizer):
        """Encode text to sequences"""
        sequences = tokenizer.texts_to_sequences(texts)
        padded = keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.max_length
        )
        return padded
```

### Sequence-to-Sequence (Seq2Seq)

```python
class Seq2SeqModel:
    """Encoder-decoder for sequence generation"""
    
    def __init__(self, input_vocab_size, output_vocab_size, latent_dim=256):
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.latent_dim = latent_dim
        self.encoder, self.decoder, self.model = self._build_model()
    
    def _build_model(self):
        """Build seq2seq architecture"""
        # Encoder
        encoder_inputs = layers.Input(shape=(None,))
        encoder_embedding = layers.Embedding(
            self.input_vocab_size,
            self.latent_dim
        )(encoder_inputs)
        encoder_lstm = layers.LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.Input(shape=(None,))
        decoder_embedding = layers.Embedding(
            self.output_vocab_size,
            self.latent_dim
        )(decoder_inputs)
        decoder_lstm = layers.LSTM(
            self.latent_dim,
            return_sequences=True,
            return_state=True
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding,
            initial_state=encoder_states
        )
        decoder_dense = layers.Dense(self.output_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Model
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        return encoder, decoder_lstm, model
```

## 💡 Interview Talking Points

**Q: What's the difference between LSTM and GRU?**
```
Answer:
- LSTM: 3 gates (input, forget, output), more parameters
- GRU: 2 gates (reset, update), simpler, faster
- Performance similar, GRU preferred for real-time/mobile
```

**Q: How handle vanishing gradient problem?**
```
Answer:
- Gradient clipping
- Initialize weights properly
- LSTM/GRU design (gates help preserve gradients)
- Use better optimizers (Adam vs SGD)
```

## 🌟 Portfolio Value

✅ LSTM/GRU architectures
✅ Time-series forecasting
✅ Text classification
✅ Seq2Seq models
✅ RNN fundamentals
✅ Deep learning expertise

---

**Technologies**: TensorFlow, Keras, NumPy

