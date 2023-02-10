from keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model
import numpy as np
import utils as u

m = 10000
dataset, human_vocab, machine_vocab, inv_machine = u.load_dataset(m)

Tx = 30
Ty = 10
X, Y, X_oh, Y_oh = u.preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("X_oh.shape:", X_oh.shape)
print("Y_oh.shape:", Y_oh.shape)

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", X_oh[index])
print("Target after preprocessing (one-hot):", Y_oh[index])

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis = -1)
densor1 = Dense(10, activation = 'tanh')
densor2 = Dense(1, activation = 'relu')
activator = Activation(u.softmax, name = 'attention_weights')
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(inputs = concat)
    energies = densor2(inputs = e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    
    return context

n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state 's'

post_activation_lstm_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation = u.softmax)

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    X = Input((Tx, human_vocab_size))
    s0 = Input(shape = (n_s, ), name = 's0')
    c0 = Input(shape = (n_s, ), name = 'c0')
    s = s0
    c = c0
    outputs = []
    
    a = Bidirectional(LSTM(units = n_a, return_sequences = True))(X)
    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_activation_lstm_cell(inputs = context, initial_state = [s, c])
        out = output_layer(inputs = s)
        outputs.append(out)
    
    model = Model(inputs = [X, s0, c0], outputs = outputs)
    
    return model

model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

opt = Adam(lr = 0.005, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Y_oh.swapaxes(0, 1))

model.fit([X_oh, s0, c0], outputs, epochs = 1, batch_size = 100)

model.load_weights('models/model.h5')

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    source = u.string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes = len(human_vocab)), source)))
    source = np.expand_dims(source, axis = 0)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine[int(i)] for i in prediction]
    
    print('source:', example)
    print('output:', ''.join(output), "\n")