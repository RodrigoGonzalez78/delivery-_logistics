from tensorflow import keras


def create_model(input_shape, learning_rate=0.001):
   
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        
        # Arquitectura Profunda
        keras.layers.Dense(128, activation='relu'), 
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mse']
    )
    return model