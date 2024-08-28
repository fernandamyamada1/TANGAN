tones = [31, 255, 62, 155, 124, 93, 186]

image_shape = 512


def wmae(y_true, y_pred, penalty_factor=5):
    # Calculate absolute differences
    absolute_diff = tf.abs(y_true - y_pred)

    # Apply penalty factor where y_true is 0
    penalty = tf.where(tf.equal(y_true, 0), penalty_factor * absolute_diff, absolute_diff)

    # Calculate mean
    return tf.reduce_mean(penalty)

def calculate_hu_moments(grayscale_image):

    # Convert the grayscale image to float32
    grayscale_image = tf.cast(grayscale_image, dtype=tf.float32)

    # Normalize the grayscale image
    mean = tf.reduce_mean(grayscale_image)
    std = tf.math.reduce_std(grayscale_image)
    normalized_image = (grayscale_image - mean) / (std + 1e-10)  # Add a small value to avoid division by zero

    # Calculate the image moments
    x = tf.range(tf.shape(normalized_image)[1], dtype=tf.float32)
    y = tf.range(tf.shape(normalized_image)[0], dtype=tf.float32)
    x, y = tf.meshgrid(x, y)
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    flattened_image = tf.reshape(normalized_image, [-1])
    m00 = tf.reduce_sum(flattened_image)
    m10 = tf.reduce_sum(x * flattened_image)
    m01 = tf.reduce_sum(y * flattened_image)
    m20 = tf.reduce_sum(x ** 2 * flattened_image)
    m02 = tf.reduce_sum(y ** 2 * flattened_image)
    m11 = tf.reduce_sum(x * y * flattened_image)
    m30 = tf.reduce_sum(x ** 3 * flattened_image)
    m03 = tf.reduce_sum(y ** 3 * flattened_image)
    m12 = tf.reduce_sum(x * y ** 2 * flattened_image)
    m21 = tf.reduce_sum(x ** 2 * y * flattened_image)

    # Calculate the Hu Moments
    hu1 = m20 + m02
    hu2 = (m20 - m02) ** 2 + 4 * m11 ** 2
    hu3 = (m30 - 3 * m12) ** 2 + (3 * m21 - m03) ** 2
    hu4 = (m30 + m12) ** 2 + (m21 + m03) ** 2
    hu5 = (m30 - 3 * m12) * (m30 + m12) * ((m30 + m12) ** 2 - 3 * (m21 + m03) ** 2) + \
          (3 * m21 - m03) * (m21 + m03) * (3 * (m30 + m12) ** 2 - (m21 + m03) ** 2)
    hu6 = (m20 - m02) * ((m30 + m12) ** 2 - (m21 + m03) ** 2) + \
          4 * m11 * (m30 + m12) * (m21 + m03)
    hu7 = (3 * m21 - m03) * (m30 + m12) * ((m30 + m12) ** 2 - 3 * (m21 + m03) ** 2) - \
          (m30 - 3 * m12) * (m21 + m03) * (3 * (m30 + m12) ** 2 - (m21 + m03) ** 2)

    # Return the Hu Moments as a tensor
    hu_moments = tf.stack([hu1, hu2, hu3, hu4, hu5, hu6])

    return hu_moments



def wmae_hu_loss(y_true, y_pred):

    # Calculate LossCWMAE
    data = tf.stack([y_true, y_pred], axis=1)
    sum = tf.math.divide(tf.math.add(sum, wmae(y_true, y_pred)) * 0.5, batch)

    for i in data:
        pred = i[0]
        true = i[1]

        for tone in tones:

            # Calculate the number of pixels for each tone
            tone = tf.constant(tone/255, dtype=tf.float32)
            mask_pred = tensor_threshold(tone, pred)
            n_pred = tf.reduce_sum(mask_pred)
            mask_true = tensor_threshold(tone, true)
            n_true = tf.reduce_sum(mask_true)

            # Calculate Hu Moments
            hu_pred = calculate_hu_moments(mask_pred)
            hu_true = calculate_hu_moments(mask_true)

            # Calculate the ratio of correct pixels for each tone
            n_true = tf.cast(n_true, tf.float32)
            n_pred = tf.cast(n_pred, tf.float32)
            diff = tf.math.abs(tf.math.subtract(n_pred, n_true))
            ratio = diff / (diff + 1) 
            ratio = tf.cast(ratio, tf.float32)

            # Correction for Nan and Inf values
            hu_true = tf.where(tf.math.is_nan(hu_true), tf.zeros_like(hu_true), hu_true)
            hu_true = tf.where(tf.math.is_inf(hu_true), tf.zeros_like(hu_true), hu_true)
            hu_pred = tf.where(tf.math.is_nan(hu_pred), tf.zeros_like(hu_pred), hu_pred)
            hu_pred = tf.where(tf.math.is_inf(hu_pred), tf.zeros_like(hu_pred), hu_pred)

            # Calculate LossHu
            dist = tf.abs(tf.subtract(hu_true, hu_pred))
            dist = tf.cast(dist, tf.float32)
            d1 = dist + tf.ones_like(dist)
            dist = dist / d1
            dist = tf.reduce_sum(dist) * ratio

            # Return LossWMAE-Hu
            sum = tf.math.add(sum, tf.math.divide(dist * 0.5, (len(tones) * batch * 6)))
    return sum


def discriminator(image_shape):

    init = RandomNormal(stddev=0.02)
    in_src_image = tf.keras.layers.Input(shape=image_shape)
    in_target_image = tf.keras.layers.Input(shape=image_shape)
    merged = tf.keras.layers.Concatenate()([in_src_image, in_target_image])

    d = tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)

    d = tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = tf.keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = tf.keras.layers.Activation('sigmoid')(d)
    model = tf.keras.models.Model([in_src_image, in_target_image], patch_out)

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, loss_weights=[0.5])
    
    return model


class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim, 1, 2))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



def model(input_shape):

    latent_dim = 2

    input_layer = tf.keras.layers.Input(shape=input_shape)
    x1 = tf.keras.layers.Conv2D(48, kernel_size=(11, 11), strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(input_layer)
    x1 = tf.keras.layers.BatchNormalization(axis=-1)(x1)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1)

    x1_0 = tf.keras.layers.Conv2D(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x1)
    x1_0 = tf.keras.layers.Dropout(.2)(x1_0)
    x1_0 = tf.keras.layers.BatchNormalization(axis=-1)(x1_0)
    x1_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1_0)

    x2 = tf.keras.layers.Conv2D(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x1_0)
    x2 = tf.keras.layers.Dropout(.2)(x2)
    x2 = tf.keras.layers.BatchNormalization(axis=-1)(x2)
    x2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2)

    x2_0 = tf.keras.layers.Conv2D(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x2)
    x2_0 = tf.keras.layers.Dropout(.2)(x2_0)
    x2_0 = tf.keras.layers.BatchNormalization(axis=-1)(x2_0)
    x2_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2_0)

    x3 = tf.keras.layers.Conv2D(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x2_0)
    x3 = tf.keras.layers.Dropout(.2)(x3)
    x3 = tf.keras.layers.BatchNormalization(axis=-1)(x3)
    x3 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3)

    x3_0 = tf.keras.layers.Conv2D(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x3)
    x3_0 = tf.keras.layers.Dropout(.2)(x3_0)
    x3_0 = tf.keras.layers.BatchNormalization(axis=-1)(x3_0)
    x3_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3_0)

    x4 = tf.keras.layers.Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x3_0)
    x4 = tf.keras.layers.Dropout(.2)(x4)
    x4 = tf.keras.layers.BatchNormalization(axis=-1)(x4)
    x4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4)

    x4_0 = tf.keras.layers.Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x4)
    x4_0 = tf.keras.layers.Dropout(.2)(x4_0)
    x4_0 = tf.keras.layers.BatchNormalization(axis=-1)(x4_0)
    x4_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4_0)

    x5 = tf.keras.layers.Conv2D(48, kernel_size=(3, 3), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(x4_0)
    x5 = tf.keras.layers.BatchNormalization(axis=-1)(x5)
    x5 = tf.keras.layers.LeakyReLU(alpha=0.2)(x5)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x5)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x5)
    z = Sampling()([z_mean, z_log_var])


    tr1 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(z)
    tr1 = tf.keras.layers.BatchNormalization(axis=-1)(tr1)
    tr1 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr1)

    tr2 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr1, x4_0]))
    tr2 = tf.keras.layers.Dropout(.2)(tr2)
    tr2 = tf.keras.layers.BatchNormalization(axis=-1)(tr2)
    tr2 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr2)

    tr2_0 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(7, 7), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr2, x4]))
    tr2_0 = tf.keras.layers.Dropout(.2)(tr2_0)
    tr2_0 = tf.keras.layers.BatchNormalization(axis=-1)(tr2_0)
    tr2_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr2_0)

    tr3 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr2_0, x3_0]))
    tr3 = tf.keras.layers.Dropout(.2)(tr3)
    tr3 = tf.keras.layers.BatchNormalization(axis=-1)(tr3)
    tr3 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr3)

    tr3_0 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(9, 9), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr3, x3]))
    tr3_0 = tf.keras.layers.Dropout(.2)(tr3_0)
    tr3_0 = tf.keras.layers.BatchNormalization(axis=-1)(tr3_0)
    tr3_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr3_0)

    tr4 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(11, 11), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr3_0, x2_0]))
    tr4 = tf.keras.layers.Dropout(.2)(tr4)
    tr4 = tf.keras.layers.BatchNormalization(axis=-1)(tr4)
    tr4 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr4)

    tr4_0 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(11, 11), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr4, x2]))
    tr4_0 = tf.keras.layers.Dropout(.2)(tr4_0)
    tr4_0 = tf.keras.layers.BatchNormalization(axis=-1)(tr4_0)
    tr4_0 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr4_0)


    tr5 = tf.keras.layers.Conv2DTranspose(48, kernel_size=(11, 11), strides=(2,2), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr4_0, x1_0]))
    tr5 = tf.keras.layers.Dropout(.2)(tr5)
    tr5 = tf.keras.layers.BatchNormalization(axis=-1)(tr5)
    tr5 = tf.keras.layers.LeakyReLU(alpha=0.2)(tr5)

    output = tf.keras.layers.Conv2D(input_shape[2], kernel_size=(1, 1), strides=(1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2())(tf.keras.layers.Concatenate()([tr5, x1]))
    output = tf.keras.layers.BatchNormalization(axis=-1)(output)
    output = tf.keras.layers.Activation("softplus")(output)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output])
    model.compile()
    return model