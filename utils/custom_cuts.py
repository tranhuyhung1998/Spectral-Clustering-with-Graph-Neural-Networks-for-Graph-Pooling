import tensorflow as tf
from keras import backend as K, activations
from keras import regularizers, constraints, initializers
from keras.engine import Layer

from spektral.layers import ops, filter_dot, MinCutPool


class TraceMinCutPool(MinCutPool):
    def call(self, inputs):
        # Note that I is useless, because thee layer cannot be used in graph
        # batch mode.
        if len(inputs) == 3:
            X, A, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X, A = inputs
            I = None

        # Check if the layer is operating in batch mode (X and A have rank 3)
        batch_mode = K.ndim(A) == 3

        # Optionally compute hidden layer
        if self.h is None:
            Hid = X
        else:
            Hid = K.dot(X, self.kernel_in)
            if self.use_bias:
                Hid = K.bias_add(Hid, self.bias_in)
            if self.activation is not None:
                Hid = self.activation(Hid)

        # Compute cluster assignment matrix
        S = K.dot(Hid, self.kernel_out)
        if self.use_bias:
            S = K.bias_add(S, self.bias_out)
        S = activations.softmax(S, axis=-1)  # Apply softmax to get cluster assignments

        # MinCut regularization
        A_pooled = ops.matmul_AT_B_A(S, A)
        D = ops.degree_matrix(A)
        cut_loss = tf.trace(ops.matmul_A_B(tf.matrix_inverse(ops.matmul_AT_B_A(S, D)), A_pooled))

        # den = tf.trace(ops.matmul_AT_B_A(S, D))
        # cut_loss = -(num / den)
        if batch_mode:
            cut_loss = K.mean(cut_loss)
        self.add_loss(cut_loss)

        # Orthogonality regularization
        SS = ops.matmul_AT_B(S, S)
        I_S = tf.eye(self.k)
        ortho_loss = tf.norm(
            SS / tf.norm(SS, axis=(-1, -2)) - I_S / tf.norm(I_S), axis=(-1, -2)
        )
        if batch_mode:
            ortho_loss = K.mean(cut_loss)
        self.add_loss(ortho_loss)

        # Pooling
        X_pooled = ops.matmul_AT_B(S, X)
        A_pooled = tf.linalg.set_diag(A_pooled, tf.zeros(K.shape(A_pooled)[:-1]))  # Remove diagonal
        A_pooled = ops.normalize_A(A_pooled)

        output = [X_pooled, A_pooled]

        if I is not None:
            I_mean = tf.segment_mean(I, I)
            I_pooled = ops.repeat(I_mean, tf.ones_like(I_mean) * self.k)
            output.append(I_pooled)

        if self.return_mask:
            output.append(S)

        return output