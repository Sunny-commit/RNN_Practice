# 🤖 RNN Practice - Recurrent Neural Networks

A **comprehensive guide to RNNs** including LSTM, GRU, sequence-to-sequence models, and attention mechanisms for sequential data.

## 🎯 Overview

This project provides:
- ✅ RNN fundamentals
- ✅ LSTM & GRU architectures
- ✅ Sequence-to-sequence models
- ✅ Attention mechanisms
- ✅ Time series prediction
- ✅ Language modeling
- ✅ Bidirectional RNNs

## 🔄 Vanilla RNN

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Input, Dense
from tensorflow.keras.models import Sequential

class VanillaRNN:
    """Basic RNN implementation"""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, X):
        """Forward pass"""
        batch_size, seq_len, _ = X.shape
        h = np.zeros((batch_size, self.hidden_size))
        
        hidden_states = []
        outputs = []
        
        for t in range(seq_len):
            x_t = X[:, t, :]  # Current input
            
            # Hidden state update
            h_t = np.tanh(
                np.dot(x_t, self.Wxh.T) + 
                np.dot(h, self.Whh.T) + 
                self.bh.T
            )
            
            # Output
            y_t = np.dot(h_t, self.Why.T) + self.by.T
            
            hidden_states.append(h_t)
            outputs.append(y_t)
            h = h_t
        
        return np.array(outputs), np.array(hidden_states)

# Using Keras
model = Sequential([
    SimpleRNN(128, input_shape=(None, input_size), return_sequences=True),
    SimpleRNN(64),
    Dense(output_size, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

## 📦 LSTM Architecture

```python
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential

class LSTMNetwork:
    """LSTM implementation"""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
    
    @staticmethod
    def build_model(input_shape, hidden_size=128, output_size=1, num_layers=2):
        """Build LSTM model"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=hidden_size,
            input_shape=input_shape,
            return_sequences=(num_layers > 1)
        ))
        model.add(Dropout(0.2))
        
        # Additional LSTM layers
        for _ in range(num_layers - 1):
            model.add(LSTM(
                units=hidden_size,
                return_sequences=(_ < num_layers - 2)
            ))
            model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(output_size, activation='linear'))
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @staticmethod
    def explain_gates():
        """LSTM gate mechanics"""
        print("""
        LSTM Gates:
        
        1. Input Gate (i_t):
           i_t = σ(W_ii * x_t + W_hi * h_t-1 + b_i)
           Controls what new information to add
        
        2. Forget Gate (f_t):
           f_t = σ(W_if * x_t + W_hf * h_t-1 + b_f)
           Controls what to forget from previous cell state
        
        3. Cell Gate (g_t):
           g_t = tanh(W_ig * x_t + W_hg * h_t-1 + b_g)
           New candidate value
        
        4. Output Gate (o_t):
           o_t = σ(W_io * x_t + W_ho * h_t-1 + b_o)
           Controls what to output
        
        5. Cell State Update:
           C_t = f_t ⊙ C_t-1 + i_t ⊙ g_t
        
        6. Hidden State Update:
           h_t = o_t ⊙ tanh(C_t)
        """)

# Train LSTM
X_train = np.random.randn(1000, 50, 10)  # (samples, timesteps, features)
y_train = np.random.randn(1000, 1)

model = LSTMNetwork.build_model(
    input_shape=(50, 10),
    hidden_size=128,
    output_size=1,
    num_layers=2
)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

## 🎯 GRU (Gated Recurrent Unit)

```python
from tensorflow.keras.layers import GRU

class GRUNetwork:
    """GRU implementation"""
    
    @staticmethod
    def build_model(input_shape, hidden_size=128, output_size=1):
        """Build GRU model"""
        model = Sequential([
            GRU(hidden_size, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            GRU(hidden_size),
            Dropout(0.2),
            Dense(output_size)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    @staticmethod
    def gru_vs_lstm():
        """Comparison"""
        print("""
        GRU vs LSTM:
        
        GRU (2 gates):
        - Reset gate: decides what to forget
        - Update gate: decides what to update
        - Simpler than LSTM
        - Faster training
        
        LSTM (3 gates):
        - Input, forget, output gates
        - More expressiveness
        - Better for longer sequences
        - More parameters to train
        
        Rule of thumb:
        - Use GRU for simpler problems
        - Use LSTM for complex sequences
        """)
```

## 🔀 Sequence-to-Sequence

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed

class Seq2SeqModel:
    """Sequence-to-sequence architecture"""
    
    @staticmethod
    def build_autoencoder(input_length, output_length, encoder_hidden=256):
        """Encoder-decoder architecture"""
        
        # Encoder
        encoder_input = Input(shape=(input_length, 1))
        encoder = LSTM(encoder_hidden, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_input)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_input = Input(shape=(output_length, 1))
        decoder = LSTM(encoder_hidden, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder(decoder_input, initial_state=encoder_states)
        
        # Dense layer for output
        dense = Dense(1)
        decoder_outputs = dense(decoder_outputs)
        
        # Model
        model = Model([encoder_input, decoder_input], decoder_outputs)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    @staticmethod
    def build_inference_encoder(model):
        """Extract encoder for inference"""
        encoder_inputs = model.input[0]
        encoder_lstm = model.layers[1]
        
        encoder_model = Model(encoder_inputs, encoder_lstm.states)
        return encoder_model
```

## 💡 Attention Mechanism

```python
from tensorflow.keras.layers import Layer
import tensorflow as tf

class AttentionLayer(Layer):
    """Attention mechanism"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W1 = None
        self.W2 = None
        self.b = None
        self.V = None
    
    def build(self, input_shape):
        """Initialize weights"""
        # input_shape: [(batch, seq_len, hidden), (batch, hidden)]
        self.W1 = self.add_weight(
            shape=(input_shape[0][-1], input_shape[0][-1]),
            initializer='glorot_uniform'
        )
        self.W2 = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer='glorot_uniform'
        )
        self.b = self.add_weight(
            shape=(input_shape[0][-1],),
            initializer='zeros'
        )
        self.V = self.add_weight(
            shape=(1, input_shape[0][-1]),
            initializer='glorot_uniform'
        )
        super().build(input_shape)
    
    def call(self, inputs):
        """Compute attention"""
        encoder_outputs, decoder_state = inputs
        
        # Calculate scores
        scores = tf.nn.tanh(
            tf.matmul(encoder_outputs, tf.transpose(self.W1)) +
            tf.matmul(tf.expand_dims(decoder_state, 1), tf.transpose(self.W2)) +
            self.b
        )
        
        # Attention weights
        attention_weights = tf.nn.softmax(
            tf.matmul(scores, tf.transpose(self.V)),
            axis=1
        )
        
        # Context vector
        context = tf.reduce_sum(
            encoder_outputs * attention_weights,
            axis=1
        )
        
        return context, attention_weights

# Build attention model
def build_attention_seq2seq(encoder_hidden=256, decoder_hidden=256):
    """Seq2seq with attention"""
    encoder_input = Input(shape=(None, 10))
    encoder = LSTM(encoder_hidden, return_sequences=True)
    encoder_outputs = encoder(encoder_input)
    
    decoder_input = Input(shape=(None, 10))
    decoder = LSTM(decoder_hidden, return_sequences=True)
    decoder_outputs = decoder(decoder_input)
    
    attention = AttentionLayer()
    context, weights = attention([encoder_outputs, decoder_outputs[:, -1, :]])
    
    output = Dense(1)(context)
    
    model = Model([encoder_input, decoder_input], output)
    return model
```

## 📊 Bidirectional RNN

```python
from tensorflow.keras.layers import Bidirectional

class BidirectionalRNN:
    """Process sequence both directions"""
    
    @staticmethod
    def build_bidirectional_lstm(input_shape, hidden_size=128):
        """Bidirectional processing"""
        model = Sequential([
            Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape=input_shape),
            Bidirectional(LSTM(hidden_size)),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
```

## 💡 Interview Talking Points

**Q: LSTM vanishing gradient problem solution?**
```
Answer:
- Skip connections in cell state
- Multiplicative gating prevents gradient explosion
- Cell state updates additively
- Forget gate prevents extreme changes
- Better gradient flow than vanilla RNN
```

**Q: When use attention?**
```
Answer:
- Long sequences (> 100 tokens)
- Need to focus on relevant parts
- Machine translation, summarization
- Improved performance on alignment tasks
- Computational cost trade-off
```

## 🌟 Portfolio Value

✅ RNN architectures
✅ LSTM mechanisms
✅ Sequence modeling
✅ Attention mechanisms
✅ Encoder-decoder
✅ Time series mastery
✅ Advanced deep learning

---

**Technologies**: TensorFlow, Keras, NumPy

