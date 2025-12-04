from tensorflow import keras


def create_model(input_shape, learning_rate=0.001):
   
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
    
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(32, activation='sigmoid'),
        
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mae']
    )
    return model