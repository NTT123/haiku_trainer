"""Trainer class."""

import pickle
import time
from functools import partial
from pathlib import Path
from typing import Any, Deque, List, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from jax.numpy import ndarray


class TrainerState(NamedTuple):
  params: hk.Params
  aux: hk.Params
  optim: optax.OptState
  rng: ndarray


class Trainer:
  def __init__(
      self,
      train_iter,
      val_iter,
      train_loss_fn,
      val_loss_fn,
      optimizer=optax.adam(1e-4),
      ckpt_freq: int = 1000,
      logging_freq: int = 100,
      out_dir: str = '/tmp/ckpts',
      resume: bool = False
  ):
    """Create a trainer object.

Args:
  - train_iter: an iterable object which yields a training data mini-batch.
  - val_iter: an iterable object which yields a valalidation data mini-batch.
  - train_loss_fn: a function which creates model and computes the training loss.
  - val_loss_fn: a function which creates model and computes the validation loss.
  - optimizer: an optax optimizer. For example: optax.adam.
  - ckpt_freq: checkpoint frequency.
  - logging_freq: logging frequency.
  - out_dir: output directory for saving checkpoints.
"""
    self.ckpt_freq = ckpt_freq
    self.logging_freq = logging_freq
    self.optimizer = optimizer
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.train_iter = train_iter
    self.train_losses = Deque(maxlen=1000)
    self.val_iter = val_iter
    self.val_losses = Deque(maxlen=100)
    self.last_step = -1

    _train_loss_fn = hk.transform_with_state(train_loss_fn)
    _vag = jax.value_and_grad(_train_loss_fn.apply, has_aux=True)
    rng = jax.random.PRNGKey(42)
    params, aux = _train_loss_fn.init(rng, next(train_iter))
    optim = optimizer.init(params)
    self.state: TrainerState = TrainerState(params, aux, optim, rng)

    def _update_ops(state: TrainerState, inputs):
      rng, rng_next = jax.random.split(state.rng, 2)
      (loss, aux), grads = _vag(state.params, state.aux, rng, inputs)
      grads, optim = optimizer.update(grads, state.optim, state.params)
      params = optax.apply_updates(state.params, grads)
      return TrainerState(params, aux, optim, rng_next), loss

    self._update_ops = jax.jit(_update_ops)

    _val_loss_obj = hk.transform_with_state(val_loss_fn)

    def _val_loss_fn(inputs, state: TrainerState):
      return _val_loss_obj.apply(state.params, state.aux, state.rng, inputs)[0]

    self._val_loss_fn = jax.jit(_val_loss_fn)

    if resume:
      self.resume()

  def resume(self):
    path = self.find_latest_checkpoint()
    if path is not None:
      print(f'loading latest checkpoint at {path}')
      assert path.name[:9] == 'hk_state_'
      self.last_step = int(path.name[9:-5], base=10)
      with open(path, 'rb') as f:
        self.load_state(f)
    else:
      print(f'No checkpoint was found at {self.out_dir}')

  def find_latest_checkpoint(self):
    ckpts = sorted(self.out_dir.glob('hk_state_*.ckpt'))
    return ckpts[-1] if len(ckpts) > 0 else None

  def run_func_with_state(self, fn):
    obj = hk.transform_with_state(fn)
    return obj.apply(self.state.params, self.state.aux, self.state.rng)[0]

  def update_ops(self, inputs):
    self.state, loss = self._update_ops(self.state, inputs)
    return loss

  def val_loss(self, inputs):
    return self._val_loss_fn(inputs, self.state)

  def load_state(self, file_obj):
    state = pickle.load(file_obj)
    self.state = jax.device_put(state)

  def save_state(self, file_obj):
    state = jax.device_get(self.state)
    pickle.dump(state, file_obj)

  def fit(self, num_steps):
    trange = tqdm.trange(self.last_step+1, num_steps,
                         initial=self.last_step+1,
                         total=num_steps,
                         desc='training',
                         ascii=True,
                         ncols=80)
    start_time = time.perf_counter()
    for step in trange:
      inputs = next(self.train_iter)
      loss = self.update_ops(inputs)
      self.train_losses.append(loss)

      if step % 10 == 0:
        inputs = next(self.val_iter)
        loss = self.val_loss(inputs)
        self.val_losses.append(loss)

      if step % self.logging_freq == 0:
        train_loss = sum(self.train_losses).item() / len(self.train_losses)
        val_loss = sum(self.val_losses).item() / len(self.val_losses)
        end_time = time.perf_counter()
        duration = end_time - start_time
        start_time = end_time
        trange.write(f'step {step}  train loss {train_loss:.3f}  val loss {val_loss:.3f}  duration {duration:.3f}')

      if step % self.ckpt_freq == 0:
        file_path = self.out_dir / f'hk_state_{step:07d}.ckpt'
        trange.write(f'saving checkpoint to file {file_path}')
        with open(file_path, 'wb') as f:
          self.save_state(f)

      self.last_step = step

    # final checkpoint
    file_path = self.out_dir / f'hk_state_{self.last_step:07d}.ckpt'
    trange.write(f'saving checkpoint to file {file_path}')
    with open(file_path, 'wb') as f:
      self.save_state(f)
