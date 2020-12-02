# haiku_trainer

A utility class for training dm-haiku models.


## Installation

```
pip install -U git+https://github.com/ntt123/haiku_trainer
```


## Example

```python

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from haiku_trainer import Trainer


def data_iter(mode='train'):
  rng = jax.random.PRNGKey(42 if mode == 'train' else 43)
  x = jax.random.normal(rng, (30, 5))
  y = jnp.sum(x, axis=1, keepdims=True) * 5 + 1.
  while True:
    yield (x, y)


def loss_fn(inputs):
  x, y = inputs
  net = hk.Linear(1)
  y_hat = net(x)
  return jnp.mean(jnp.square(y_hat - y))


trainer = Trainer(
    train_iter=data_iter('train'),
    val_iter=data_iter('val'),
    train_loss_fn=loss_fn,
    val_loss_fn=loss_fn,
    optimizer=optax.adam(1e-3),
    ckpt_freq=5000,
    logging_freq=1000,
    out_dir='/tmp/linear_regression',
    resume=False)

trainer.fit(num_steps=10_000)
```
