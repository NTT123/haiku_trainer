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

trainer = Trainer(data_iter(), data_iter(), loss_fn, loss_fn, optax.adam(1e-3))
trainer.fit(10_000)
```
