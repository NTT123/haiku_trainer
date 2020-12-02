# haiku_trainer

A utility class for training dm-haiku models.


## Installation

```
pip install -U git+https://github.com/ntt123/haiku_trainer
```


## Example

```python

from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from haiku_trainer import Trainer


def data_iter(mode='train'):
  rng = jax.random.PRNGKey(42 if mode == 'train' else 43)
  x = jax.random.normal(rng, (32, 5))
  y = jnp.sum(x, axis=1, keepdims=True) * 9. - 1.
  while True:
    yield (x, y)


def mlp(x, is_training):
  x = hk.Linear(64)(x)
  x = jax.nn.relu(x)
  x = hk.dropout(hk.next_rng_key(), 0.5, x) if is_training else x
  return hk.Linear(1)(x)


def loss_fn(inputs, is_training):
  x, y = inputs
  y_hat = mlp(x, is_training=is_training)
  return jnp.mean(jnp.square(y_hat - y))


trainer = Trainer(
    train_iter=data_iter('train'),
    val_iter=data_iter('val'),
    train_loss_fn=partial(loss_fn, is_training=True),
    val_loss_fn=partial(loss_fn, is_training=False),
    optimizer=optax.adam(1e-3),
    ckpt_freq=5000,
    logging_freq=1000,
    out_dir='/tmp/regression',
    resume=False)

trainer.fit(num_steps=10_000)

# inference
with open('/tmp/regression/hk_state_0005000.ckpt', 'rb') as f:
  trainer.load_state(f)
rng = jax.random.PRNGKey(44)
x_test = jax.random.normal(rng, (1, 5))
y_test = jnp.sum(x_test, axis=1, keepdims=True) * 9. - 1.
y_hat = trainer.run_func_with_state(partial(mlp, x_test, is_training=False))
print(y_test.T)
print(y_hat.T)
```
