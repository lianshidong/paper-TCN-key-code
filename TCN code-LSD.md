```python
# ===============#
#   TCN_model, edit by LSD 2025/11/22
# ===============#
class TCN_RES_model(DLModel):

    def constructModel(self):
        inp_seq = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]), name='EEG_seq')
        inp_feat = Input(shape=(self.X_train_feat.shape[1], self.X_train_feat.shape[2]), name='EEG_feat')

        seq_4d = Reshape((self.X_train.shape[1], 1, self.X_train.shape[2]))(inp_seq)

        ca = ChannelAttention(in_planes=self.X_train.shape[2], ratio=8)
        channel_weights = ca(seq_4d)  
        channel_refined = Multiply()([seq_4d, channel_weights]) 

        sa = SpatialAttention(kernel_size=5)
        spatial_refined = sa(channel_refined) 

        att_output = Reshape((self.X_train.shape[1], self.X_train.shape[2]))(spatial_refined)

        num_heads = self.X_train.shape[2] // 32 if self.X_train.shape[2] // 32 > 0 else 3
        key_dim = self.X_train.shape[2] // 2 if self.X_train.shape[2] // 2 > 0 else 16 
        mha_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        mha_out = mha_layer(att_output, att_output)  

        mha_flat = Flatten()(mha_out)

        x = TCN(
            nb_filters=self.conv_num * 2,
            kernel_size=5,
            dilations=(1, 2, 4),
            return_sequences=True
        )(mha_out)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        avgpool_1 = GlobalAveragePooling1D()(x) 
        maxpool_1 = GlobalMaxPooling1D()(x) 

        hidden_dim = x.shape[-1] // 8 if x.shape[-1] // 8 > 0 else 1
        avg_fc = Dense(hidden_dim, activation='relu')(avgpool_1)
        avg_fc = Dense(x.shape[-1], activation='sigmoid')(avg_fc)

        max_fc = Dense(hidden_dim, activation='relu')(maxpool_1)
        max_fc = Dense(x.shape[-1], activation='sigmoid')(max_fc)

        ch_att_1 = Multiply()([avg_fc, max_fc]) 
        ch_att_1 = Reshape((1, ch_att_1.shape[-1]))(ch_att_1)
        x = Multiply()([x, ch_att_1]) 

        tau_1 = self.get_threshold_global(x)  
        tau_1 = tf.reshape(tau_1, [-1, 1, 1])  
        tau_1 = tf.broadcast_to(tau_1, tf.shape(x)) 

        x = Lambda(lambda z: soft_threshold(z[0], z[1]))([x, tau_1])  

        x1 = AveragePooling1D(pool_size=3, padding='valid')(x)
        x1 = ReLU()(x1)

        x2 = TCN(
            nb_filters=self.conv_num * 2,
            kernel_size=3,
            dilations=(1, 3),
            return_sequences=False
        )(x1)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)

        hidden_dim2 = x2.shape[-1] // 8 if x2.shape[-1] // 8 > 0 else 1
        avg_2 = Dense(hidden_dim2, activation='relu')(x2)
        avg_2 = Dense(x2.shape[-1], activation='sigmoid')(avg_2)

        max_2 = Dense(hidden_dim2, activation='relu')(x2)
        max_2 = Dense(x2.shape[-1], activation='sigmoid')(max_2)

        ch_att_2 = Multiply()([avg_2, max_2]) 
        x2 = Multiply()([x2, ch_att_2])

        tau_2 = self.get_threshold_global_2d(x2)  
        x2 = Lambda(lambda z: soft_threshold(z[0], z[1]))([x2, tau_2])
        x2_flat = Flatten()(x2)
        feat_tcn = Dense(128)(x2_flat)
        feat_att = Dense(128)(mha_flat)
        feat_inp = Flatten()(inp_feat)
        feat_inp = Dense(128)(feat_inp)

        fusion = Concatenate()([feat_tcn, feat_att, feat_inp])
        fusion = BatchNormalization()(fusion)
        fusion = Dense(256, activation='relu', name="draw_layer")(fusion)
        fusion = Dropout(0.2)(fusion)
        fusion = Dense(64, activation='relu')(fusion)

        out = Dense(self.num_class, activation='softmax')(fusion)

        self.deepModel = Model([inp_seq, inp_feat], out)

    def get_threshold_global(self, x):

        abs_mean = tf.reduce_mean(tf.abs(x), axis=[1, 2])  
        abs_mean = tf.reshape(abs_mean, (-1, 1))  
        scale = Dense(1, activation='sigmoid')(abs_mean)  
        tau = scale * abs_mean  
        return tau  

    def get_threshold_global_2d(self, x):
        abs_mean = tf.reduce_mean(tf.abs(x), axis=1, keepdims=True) 
        scale = Dense(1, activation='sigmoid')(abs_mean)  
        tau = scale * abs_mean  
        return tau
```

