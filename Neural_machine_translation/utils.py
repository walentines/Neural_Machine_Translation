import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt

fake = Faker()
Faker.seed(12345)
random.seed(12345)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

LOCALES = ['en_US']

def load_date():
   dt = fake.date_object()
    
   try:
       human_readable = format_date(dt, format = random.choice(FORMATS), locale = 'en_US')
       human_readable = human_readable.lower()
       human_readable = human_readable.replace(',', '')
       machine_readable = dt.isoformat()
    
   except AttributeError as e:
       return None, None, None
       
   return human_readable, machine_readable, dt

def load_dataset(m):
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    
    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))
    
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k, v in inv_machine.items()}
    
    return dataset, human, machine, inv_machine

def string_to_int(string, length, vocab):
    string = string.lower()
    string = string.replace(',', '')
    
    if len(string) > length:
        string = string[:length]
    
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    
    return rep

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    X, Y = zip(*dataset)
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = np.array([string_to_int(j, Ty, machine_vocab) for j in Y])
    
    X_oh = np.array(list(map(lambda x: to_categorical(x, num_classes = len(human_vocab)), X)))
    Y_oh = np.array(list(map(lambda x: to_categorical(x, num_classes = len(machine_vocab)), Y)))
    
    return X, Y, X_oh, Y_oh

def softmax(x, axis = 1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis = axis, keepdims = True))
        s = K.sum(e, axis = axis, keepdims = True)
        return e / s
    
    else:
        raise ValueError('Cannot apply softmax function to a tensor that is 1D')

def int_to_string(ints, inv_machine_vocab):
    l = [inv_machine_vocab[i] for i in ints]
    
    return l

def plot_attention_map(model, input_vocabulary, inv_output_vocabulary, text, n_s = 128, num = 6, Tx = 30, Ty = 10):
    attention_map = np.zeros((10, 30))
    Ty, Tx = attention_map.shape
    
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    layer = model.layers[num]

    encoded = np.array(string_to_int(text, Tx, input_vocabulary)).reshape((1, 30))
    encoded = np.array(list(map(lambda x: to_categorical(x, num_classes=len(input_vocabulary)), encoded)))

    f = K.function(model.inputs, [layer.get_output_at(t) for t in range(Ty)])
    r = f([encoded, s0, c0])
    
    for t in range(Ty):
        for t_prime in range(Tx):
            attention_map[t][t_prime] = r[t][0,t_prime,0]

    prediction = model.predict([encoded, s0, c0])
    
    predicted_text = []
    for i in range(len(prediction)):
        predicted_text.append(int(np.argmax(prediction[i], axis=1)))
        
    predicted_text = list(predicted_text)
    predicted_text = int_to_string(predicted_text, inv_output_vocabulary)
    text_ = list(text)
    
    input_length = len(text)
    output_length = Ty
    
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)

    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')

    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text[:output_length])

    ax.set_xticks(range(input_length))
    ax.set_xticklabels(text_[:input_length], rotation=45)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    ax.grid()
    
    return attention_map