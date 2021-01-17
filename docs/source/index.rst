:github_url: https://github.com/ntt123/haiku_trainer/tree/master/docs

haiku_trainer Documentation
===========================

haiku_trainer is a simple python class for training ``dm-haiku`` machine learning models.

.. code-block:: python

    from functools import partial
    import haiku as hk
    import jax
    import jax.numpy as jnp
    import optax

    from haiku_trainer import Trainer

    # fake data iterator
    def data_iter(mode='train'):
      rng = jax.random.PRNGKey(42 if mode == 'train' else 43)
      x = jax.random.normal(rng, (32, 5))
      y = jnp.sum(x, axis=1, keepdims=True) * 9. - 1.
      while True:
        yield (x, y)

    # MLP haiku model
    def mlp(x, is_training):
      x = hk.Linear(64)(x)
      x = jax.nn.relu(x)
      x = hk.dropout(hk.next_rng_key(), 0.5, x) if is_training else x
      return hk.Linear(1)(x)

    #  mse loss function
    def loss_fn(inputs, is_training):
      x, y = inputs
      y_hat = mlp(x, is_training=is_training)
      return jnp.mean(jnp.square(y_hat - y))

    # create trainer object
    trainer = Trainer(
        train_loss_fn=partial(loss_fn, is_training=True),
        train_data_iter=data_iter('train'),
        val_loss_fn=partial(loss_fn, is_training=False),
        val_data_iter=data_iter('val'),
        optimizer=optax.adam(1e-3),
        ckpt_freq=5000,
        logging_freq=1000,
        out_dir='/tmp/regression',
        resume=False)
    # train model
    trainer.fit(total_steps=10_000)



Installation
------------

To install the latest version of haiku_trainer, run::

    $ pip install git+https://github.com/ntt123/haiku_trainer

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :caption: API Documentation
   :maxdepth: 1

   api



License
-------

haiku_trainer is licensed under the MIT License.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
