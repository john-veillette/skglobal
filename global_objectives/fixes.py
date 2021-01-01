from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_decorator
from tensorflow.python.framework import dtypes
import functools

_ARGSTACK = [{}]
_DECORATED_OPS = {}

def add_arg_scope(func):
  """Decorates a function with args so it can be used within an arg_scope.
  Args:
    func: function to decorate.
  Returns:
    A tuple with the decorated function func_with_args().
  """

  def func_with_args(*args, **kwargs):
    current_scope = current_arg_scope()
    current_args = kwargs
    key_func = arg_scope_func_key(func)
    if key_func in current_scope:
      current_args = current_scope[key_func].copy()
      current_args.update(kwargs)
    return func(*args, **current_args)

  _add_op(func)
  setattr(func_with_args, '_key_op', arg_scope_func_key(func))
  return tf_decorator.make_decorator(func, func_with_args)

def current_arg_scope():
  stack = _get_arg_stack()
  return stack[-1]

def _get_arg_stack():
  if _ARGSTACK:
    return _ARGSTACK
  else:
    _ARGSTACK.append({})
    return _ARGSTACK

def arg_scope_func_key(op):
  return getattr(op, '_key_op', str(op))

def _add_op(op):
  key = arg_scope_func_key(op)
  if key not in _DECORATED_OPS:
    _DECORATED_OPS[key] = _kwarg_names(op)

def _kwarg_names(func):
  kwargs_length = len(func.__defaults__) if func.__defaults__ else 0
  return func.__code__.co_varnames[-kwargs_length:func.__code__.co_argcount]


@add_arg_scope
def model_variable(name, shape=None, dtype=dtypes.float32, initializer=None,
                   regularizer=None, trainable=True, collections=None,
                   caching_device=None, device=None, partitioner=None,
                   custom_getter=None, use_resource=None):
  """Gets an existing model variable with these parameters or creates a new one.
  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    regularizer: a (Tensor -> Tensor or None) function; the result of
        applying it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: A list of collection names to which the Variable will be added.
      Note that the variable is always also added to the
      `GraphKeys.GLOBAL_VARIABLES` and `GraphKeys.MODEL_VARIABLES` collections.
    caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and dtype of the `Variable` to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    custom_getter: Callable that allows overwriting the internal
      get_variable method and has to have the same signature.
    use_resource: If `True` use a ResourceVariable instead of a Variable.
  Returns:
    The created or existing variable.
  """
  collections = list(collections or [])
  collections += [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES]
  var = variable(name, shape=shape, dtype=dtype,
                 initializer=initializer, regularizer=regularizer,
                 trainable=trainable, collections=collections,
                 caching_device=caching_device, device=device,
                 partitioner=partitioner, custom_getter=custom_getter,
                 use_resource=use_resource)
  return var


@add_arg_scope
def variable(name, shape=None, dtype=None, initializer=None,
             regularizer=None, trainable=True, collections=None,
             caching_device=None, device=None,
             partitioner=None, custom_getter=None, use_resource=None):
  """Gets an existing variable with these parameters or creates a new one.
  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    regularizer: a (Tensor -> Tensor or None) function; the result of
        applying it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: A list of collection names to which the Variable will be added.
      If None it would default to `tf.GraphKeys.GLOBAL_VARIABLES`.
    caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and dtype of the `Variable` to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    custom_getter: Callable that allows overwriting the internal
      get_variable method and has to have the same signature.
    use_resource: If `True` use a ResourceVariable instead of a Variable.
  Returns:
    The created or existing variable.
  """
  collections = list(collections if collections is not None
                     else [ops.GraphKeys.GLOBAL_VARIABLES])

  # Remove duplicates
  collections = list(set(collections))
  getter = variable_scope.get_variable
  if custom_getter is not None:
    getter = functools.partial(custom_getter,
                               reuse=variable_scope.get_variable_scope().reuse)
  with ops.device(device or ''):
    return getter(name, shape=shape, dtype=dtype,
                  initializer=initializer,
                  regularizer=regularizer,
                  trainable=trainable,
                  collections=collections,
                  caching_device=caching_device,
                  partitioner=partitioner,
                  use_resource=use_resource)
	






