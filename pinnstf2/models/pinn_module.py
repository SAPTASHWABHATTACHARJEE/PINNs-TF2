from typing import List, Dict, Callable, Any, Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp
import sys, os, logging, time

from pinnstf2.utils import fwd_gradient, gradient
from pinnstf2.utils import (
    fix_extra_variables,
    mse,
    relative_l2_error,
    sse
)

class PINNModule:
    def __init__(
        self,
        net,
        pde_fn,
        loss_fn: str = "sse",
        extra_variables=None,
        output_fn=None,
        runge_kutta=None,
        jit_compile: bool = True,
        dtype: str = 'float32'
    ) -> None:
        super().__init__()
        
        self.net = net
        self.tf_dtype = tf.as_dtype(dtype)

        if hasattr(self.net, 'model'):
            self.trainable_variables = self.net.model.trainable_variables
        else:
            self.trainable_variables = self.net.trainable_variables

        (self.trainable_variables,
         self.extra_variables) = fix_extra_variables(self.trainable_variables, extra_variables, self.tf_dtype)

        self.output_fn = output_fn
        self.rk = runge_kutta
        self.pde_fn = pde_fn

        if loss_fn == "sse":
            self.loss_fn = sse
        elif loss_fn == "mse":
            self.loss_fn = mse
        else:
            raise ValueError("Unsupported loss function")

        # ✅ Replace Adam with L-BFGS
        self.opt = None   # L-BFGS will not be a tf.keras optimizer

        if jit_compile:
            self.train_step = tf.function(self.train_step, jit_compile=True)
            self.eval_step = tf.function(self.eval_step, jit_compile=True)
        else:
            self.train_step = tf.function(self.train_step, jit_compile=False)
            self.eval_step = tf.function(self.eval_step, jit_compile=False)

    # helper to flatten/unflatten variables for L-BFGS
    def _pack_variables(self):
        return tf.dynamic_stitch(
            [tf.range(tf.size(v)) for v in self.trainable_variables],
            [tf.reshape(v, [-1]) for v in self.trainable_variables]
        )

    def _unpack_variables(self, vector):
        idx = 0
        for v in self.trainable_variables:
            size = tf.size(v)
            new_val = tf.reshape(vector[idx: idx + size], v.shape)
            v.assign(new_val)
            idx += size

    def train_step(self, batch, max_iterations=500, tolerance=1e-9):
        """Train with L-BFGS instead of Adam"""

        def value_and_gradients_fn(var_vector):
            self._unpack_variables(var_vector)
            with tf.GradientTape() as tape:
                loss, _ = self.model_step(batch)
            grads = tape.gradient(loss, self.trainable_variables)
            grads_vector = tf.dynamic_stitch(
                [tf.range(tf.size(g)) for g in grads],
                [tf.reshape(g, [-1]) for g in grads]
            )
            return loss, grads_vector

        init_vars = self._pack_variables()

        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=value_and_gradients_fn,
            initial_position=init_vars,
            max_iterations=max_iterations,
            tolerance=tolerance
        )

        self._unpack_variables(results.position)
        return results.objective_value, self.extra_variables
