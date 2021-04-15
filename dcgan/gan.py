from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU

def discriminator():
    model = Sequential([
        Conv2D(
            filters = 64, 
            kernel_size = (3,3), 
            strides = (2,2), 
            padding = "same", 
            input_shape = (28,28,1)
        ),
        LeakyReLU(alpha = 0.2),
        Conv2D(
            filters = 64, 
            kernel_size = (3,3), 
            strides = (2,2), 
            padding = "same"
        ),
        LeakyReLU(alpha = 0.2),
        Flatten(),
        Dense(1, activation = "sigmoid")
    ])
    optimizer = Adam(lr = 0.0002, beta_1 = 0.5)
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
    return model

def generator():
    model = Sequential(
        [

        ]
    )
    model.compile()
    return model

def real_samples():
    (X_train, _), (_, _) = mnist.load_data()
    X = np.expand_dims(X_train, axis = 1)
    X = X.astype('float32')
    X = X / 255.0
    return X