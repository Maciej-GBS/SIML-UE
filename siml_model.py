import tensorflow as tf

def build_model(num_days, data_size, output_size=1) -> tf.keras.models.Model:
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=[num_days, data_size]))

    model.add(tf.keras.layers.LSTM(units=24, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.LSTM(units=12, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.LSTM(units=6, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.BatchNormalization()) # reduces quality
    model.add(tf.keras.layers.Dense(units=10, activation='relu'))
    # for regression use linear activation, i.e. no activation function is needed
    # sigmoid and other min/max value limiting functions should be avoided
    model.add(tf.keras.layers.Dense(units=output_size, activation='linear'))

    return model
