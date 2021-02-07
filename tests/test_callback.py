"""Test callback."""

from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from haiku_trainer import Trainer
from jax.util import partial


def loss_fn(inputs):
  x, y = inputs
  net = hk.Linear(1)
  y_hat = net(x)
  return jnp.sum(jnp.square(y_hat - y))


def data_iter():
  rng = jax.random.PRNGKey(42)
  while True:
    rng, rng_ = jax.random.split(rng, 2)
    x = jax.random.normal(rng_, (3, 5))
    y = jax.random.normal(rng_, (3, 1))
    yield (x, y)


def test_cb():
  trainer = Trainer(loss_fn, data_iter(), loss_fn, data_iter())

  def cb_fn(trainer: Trainer):
    net = hk.Linear(1)
    x, y = next(trainer.val_iter)
    del y
    return net(x)

  trainer.register_callback(1, cb_fn)
  trainer.fit(100)


def test_cb_jit():
  trainer = Trainer(loss_fn, data_iter(), loss_fn, data_iter())

  def cb_fn(forward: Callable, trainer: Trainer):
    x, y = next(trainer.val_iter)
    return trainer.run_func_with_state(partial(forward, x=x), do_transform=False)[0]

  def fw_fn(x): return hk.Linear(1)(x)
  fw_fn_ = hk.transform_with_state(fw_fn).apply
  fw_fn_ = jax.jit(fw_fn_)
  cb_fn_ = partial(cb_fn, forward=fw_fn_)

  trainer.register_callback(1, cb_fn_)
  trainer.fit(100)
