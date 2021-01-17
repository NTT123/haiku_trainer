"""Trainer class."""

import pickle
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Deque, List, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.numpy import ndarray
from tqdm.auto import tqdm


class TrainerState(NamedTuple):
  """TrainerState data structure.

  Attributes:
    params (hk.Params): model parameters.
    aux (hk.Params): model auxilary parameters.
    optim (optax.OptState): optimizer states.
    rng (ndarray): random states
  """
  params: hk.Params
  aux: hk.Params
  optim: optax.OptState
  rng: ndarray


class Trainer:
  """Trainer class"""

  def __init__(
      self,
      train_loss_fn,
      train_data_iter,
      val_loss_fn=None,
      val_data_iter=None,
      optimizer=optax.adam(1e-4),
      ckpt_freq: int = 1000,
      logging_freq: int = 100,
      out_dir: str = '/tmp/ckpts',
      resume: bool = False,
      wandb=None
  ):
    """Create a trainer object.

    This object will keep all the information to train a haiku model. Including: model parameters, current training step, etc.


Args:
  train_loss_fn: a function which creates model and computes the training loss.
  train_data_iter: an iterable object which yields a training data mini-batch.
  val_loss_fn: a function which creates model and computes the validation loss.
  val_data_iter: an iterable object which yields a valalidation data mini-batch.
  optimizer: an optax optimizer. For example: optax.adam.
  ckpt_freq: checkpoint frequency.
  logging_freq: logging frequency.
  out_dir: output directory for saving checkpoints.
  resume: resume the trainer to the latest checkpoint.
  wandb: wandb module object for logging (or None to disable).
"""
    self.ckpt_freq = ckpt_freq
    self.logging_freq = logging_freq
    self.optimizer = optimizer
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.train_iter = train_data_iter
    self.train_losses = Deque(maxlen=1000)
    self.val_iter = val_data_iter
    self.val_losses = Deque(maxlen=100)
    self.last_step = -1
    self.train_loss_fn = train_loss_fn
    self.val_loss_fn = val_loss_fn
    self.optimizer = optimizer
    self.compile()
    if resume:
      self.resume()
    self.start_time = time.perf_counter()
    self.wandb = wandb
    self.callbacks = []

  def register_callback(self, callback_freq: int, callback_fn: Callable):
    """Register a callback function in training process.

    Function ``callback_fn`` will be called at every ``callback_freq`` training steps.

    The callback function is called indirectly by ``Trainer.run_func_with_state``.
    This allows the function to access model states and parameters.

    Args:
      callback_freq (int): call function at `steps % callback_freq == 0`.
      callback_fn (Callable): the function to carry the task.

    """

    self.callbacks.append((callback_freq, callback_fn))

  def compile(self):
    """Compile the update function and validation function.

    The update function is used to update parameters of the network.
    It first calls the loss function, compute the gradient,
    and finally calls the optimizer.
    """
    _train_loss_fn = hk.transform_with_state(self.train_loss_fn)
    _vag = jax.value_and_grad(_train_loss_fn.apply, has_aux=True)
    rng = jax.random.PRNGKey(42)
    params, aux = _train_loss_fn.init(rng, next(self.train_iter))
    optim = self.optimizer.init(params)
    self.state: TrainerState = TrainerState(params, aux, optim, rng)

    def _update_ops(state: TrainerState, inputs):
      rng, rng_next = jax.random.split(state.rng, 2)
      (loss, aux), grads = _vag(state.params, state.aux, rng, inputs)
      grads, optim = self.optimizer.update(grads, state.optim, state.params)
      params = optax.apply_updates(state.params, grads)
      return TrainerState(params, aux, optim, rng_next), loss

    self.jit_update_ops = jax.jit(_update_ops)

    if self.val_loss_fn is not None:
      _val_loss_obj = hk.transform_with_state(self.val_loss_fn)

      def _val_loss_fn(inputs, state: TrainerState):
        return _val_loss_obj.apply(state.params, state.aux, state.rng, inputs)[0]

      self.jit_val_loss_fn = jax.jit(_val_loss_fn)

  def tiktok(self):
    """Compute the elapsed time since the last time this function is called."""
    end_time = time.perf_counter()
    duration = end_time - self.start_time
    self.start_time = end_time
    return duration

  def resume(self):
    """Find the latest checkpoint in the output directory.

    Resume the network states to the checkpoint.
    """
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
    """Return the latest checkpoint in the output directory."""
    ckpts = sorted(self.out_dir.glob('hk_state_*.ckpt'))
    return ckpts[-1] if len(ckpts) > 0 else None

  def run_func_with_state(self, fn):
    """Transform function ``fn`` and run it.

    Allows the function ``fn`` to access the model's parameters and states.
    This function is very useful when we want to run prediction after the model is trained.

    Args:
      fn(Callable): the function will be transformed.
    """
    obj = hk.transform_with_state(fn)
    return obj.apply(self.state.params, self.state.aux, self.state.rng)[0]

  def avg_training_loss(self):
    """Return the current average training loss."""
    return sum(self.train_losses).item() / len(self.train_losses)

  def avg_validation_loss(self):
    """Return the current average validation loss."""
    return sum(self.val_losses).item() / len(self.val_losses)

  def training_step(self):
    """Run one training step:
      * update network parameters,
      * and increase ``Trainer.last_step``.
    """
    inputs = next(self.train_iter)
    self.state, loss = self.jit_update_ops(self.state, inputs)
    self.last_step = self.last_step + 1
    self.train_losses.append(loss)
    return loss

  def validation_step(self):
    """Run one validation step."""
    inputs = next(self.val_iter)
    loss = self.jit_val_loss_fn(inputs, self.state)
    self.val_losses.append(loss)
    return loss

  def load_step(self, step: int):
    """Load model states at checkpoint ``step``."""
    file_path = self.out_dir / f'hk_state_{step:07d}.ckpt'
    with open(file_path, 'rb') as f:
      self.load_state(f)
    return file_path

  def load_state(self, file_obj):
    """Load model states from file object ``file_obj``."""
    state = pickle.load(file_obj)
    self.state = jax.device_put(state)

  def save_step(self, step: int):
    """Save model state at step."""
    file_path = self.out_dir / f'hk_state_{step:07d}.ckpt'
    with open(file_path, 'wb') as f:
      self.save_state(f)
    return file_path

  def save_state(self, file_obj):
    """Save model state to file objection ``file_obj``"""
    state = jax.device_get(self.state)
    pickle.dump(state, file_obj)

  def trange(self, total_steps: int):
    """Return a tqdm object of a ``range(last_step+1, total_steps)``."""
    return tqdm(range(self.last_step+1, total_steps), initial=self.last_step+1, total=total_steps, desc='training')

  def fit(self, total_steps: int = 1):
    """Fit model with the data.

    Call registered callback functions.

    Also create a checkpoint at the last training step.

    Args:
      total_steps (int): the total training steps(does not include validation step)
    """
    if self.val_loss_fn is None:
      raise ValueError('`fit` method does not support `val_loss_fn = None`')

    trange = self.trange(total_steps)
    for step in trange:
      self.training_step()

      if step % 10 == 0:
        self.validation_step()

      for callback_freq, callback_fn in self.callbacks:
        if step % callback_freq == 0:
          self.run_func_with_state(partial(callback_fn, trainer=self))

      if step % self.logging_freq == 0:
        train_loss = self.avg_training_loss()
        val_loss = self.avg_validation_loss()
        duration = self.tiktok()
        trange.write(f'step {step}  train loss {train_loss:.3f}  val loss {val_loss:.3f}  duration {duration:.3f}')
        if self.wandb is not None:
          self.wandb.log({
              'step': step,
              'train loss': train_loss,
              'val loss': val_loss,
              'duration': duration
          })

      if step % self.ckpt_freq == 0:
        file_path = self.save_step(step)
        trange.write(f'saving checkpoint to file {file_path}')

    # final checkpoint
    file_path = self.save_step(self.last_step)
    trange.write(f'saving checkpoint to file {file_path}')
