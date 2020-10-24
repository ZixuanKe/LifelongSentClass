#!/usr/bin/env python3

"""
@author: xi
@since: 2016-11-11
"""

import collections
import datetime as dt
import os

import numpy as np

from . import settings
from . import data
from . import operations
from . import widgets


class Slot(object):

    def __init__(self,
                 # session,
                 inputs=None,
                 outputs=None,
                 updates=None,
                 givens=None,
                 callbacks=None):
        """Create a Slot with the given params.

        :param inputs: Tensor or list(tuple) of Tensors.
        :param outputs: Tensor, list(tuple) of Tensors or Tensor dict.
        :param updates: Operator or list(tuple) of Operators.
        :param givens: Tensor dict.
        """
        # if session is None:
        #     raise ValueError('Invalid session.')
        # self._session = session
        self._session = settings.get_session()
        #
        # Inputs.
        if inputs is None:
            inputs = ()
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        self._inputs = inputs
        #
        # Outputs.
        if outputs is None:
            outputs = ()
        if not isinstance(outputs, (tuple, list)) \
                and not isinstance(outputs, (dict, collections.OrderedDict)):
            outputs = (outputs,)
        self._outputs = outputs
        #
        # Updates.
        if updates is None:
            updates = ()
        if not isinstance(updates, (tuple, list)):
            updates = (updates,)
        self._updates = updates
        #
        # Givens.
        if givens is None:
            givens = {}
        if not isinstance(givens, dict):
            raise ValueError('Givens must be dict.')
        self._givens = givens
        #
        # Callbacks.
        if callbacks is None:
            callbacks = ()
        if not isinstance(callbacks, (tuple, list)):
            callbacks = (callbacks,)
        self._callbacks = callbacks
        #
        self._feed_dict = givens.copy()
        self._fetches = (outputs, updates)
        if len(outputs) == 0 and len(updates) == 0:
            raise ValueError('At least one output or update should be set.')

    @property
    def outputs(self):
        return self._outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def updates(self):
        return self._updates

    @property
    def givens(self):
        return self._givens

    def __call__(self, *args):
        #
        # Check input length.
        if len(args) != len(self._inputs):
            print(len(args), len(self._inputs))
            raise ValueError('The count of parameters is not match the inputs.')
        #
        # Make "feed_dict".
        for index, placeholder in enumerate(self._inputs):
            self._feed_dict[placeholder] = args[index]
        #
        # Run the graph on the session.
        ret = self._session.run(fetches=self._fetches, feed_dict=self._feed_dict)[0]
        for callback in self._callbacks:
            callback(ret)
        return ret


class Trainer(widgets.Widget):
    """Trainer
    """

    def __init__(self,
                 name,
                 # session=None,
                 build=True):
        # if session is None:
        #     tfconfig = tf.ConfigProto()
        #     tfconfig.gpu_options.allow_growth = True
        #     self._session = tf.Session(config=tfconfig)
        #     self._session_provided = False
        # else:
        #     if not isinstance(session, tf.Session):
        #         raise ValueError('session should be tf.Session.')
        #     self._session = session
        #     self._session_provided = True
        self._slots = {}
        self._predict_slot = None
        self._fitters = collections.deque()
        super(Trainer, self).__init__(name, build)

    # def __del__(self):
    #     if not self._session_provided and self._session is not None:
    #         self._session.close()

    def _build(self):
        """Build the model.
        Abstract method.
        All subclass must implement this method.

        There are at least two tasks to be done in this method:
        1) Construct the model's graph structure with TF.
        2) Define and add slots for training, evaluation and prediction.
        """
        raise NotImplementedError()

    def _setup(self, *args, **kwargs):
        pass

    def _add_slot(self, name,
                  inputs=None,
                  outputs=None,
                  givens=None,
                  updates=None):
        if name in self._slots:
            raise ValueError('Slot {} exists.'.format(name))
        slot = Slot(
            # session=self._session,
            inputs=inputs,
            outputs=outputs,
            updates=updates,
            givens=givens
        )
        self._slots[name] = slot

    def _add_train_slot(self, inputs=None, outputs=None, givens=None, updates=None):
        self._add_slot(settings.TRAIN, inputs, outputs, givens, updates)

    def _add_validate_slot(self, inputs=None, outputs=None, givens=None, updates=None):
        self._add_slot(settings.VALIDATE, inputs, outputs, givens, updates)

    def _add_predict_slot(self, inputs=None, outputs=None, givens=None, updates=None):
        self._add_slot(settings.PREDICT, inputs, outputs, givens, updates)

    def get_slot(self, name):
        return self._slots[name] if name in self._slots else None

    def fit(self, max_loop=10000):
        """Train the model to fit the given dataset.

        :param max_loop: The number of max loop. Default is 10000.
            Here, "a loop" means train the model with one batch of data.
        """
        context = {
            settings.CONTEXT_TRAINER: self,
            settings.CONTEXT_MAX_LOOP: max_loop
        }
        for i in range(1, max_loop + 1):
            context[settings.CONTEXT_LOOP] = i
            for fitter in self._fitters:
                try:
                    fitter.fit(i, max_loop, context)
                except FitterInterrupt:
                    break

    def add_fitter(self, fitter):
        self._fitters.append(fitter)

    def clear_fitters(self):
        self._fitters.clear()

    def add_data_fitter(self, data_source, batch_size, slot_name, interval=1, count=1):
        self.add_fitter(DataFitter(data_source, batch_size, self, slot_name, interval, count))

    def add_data_trainer(self, data_source, batch_size, interval=1, count=1):
        self.add_fitter(DataFitter(data_source, batch_size, self, settings.TRAIN, interval, count))

    def add_data_validator(self, data_source, batch_size, interval=1, count=1):
        self.add_fitter(Validator(data_source, batch_size, self, settings.VALIDATE, interval, count))

    def add_screen_logger(self, log_attr, value_names=('loss',), message=None, interval=1, count=1):
        self.add_fitter(ScreenLogger(log_attr, value_names, message, interval, count))

    def predict(self, data_batch):
        if self._predict_slot is None:
            if settings.PREDICT not in self._slots:
                raise RuntimeError('No predict slot defined.')
            self._predict_slot = self._slots[settings.PREDICT]
        return self._predict_slot(*data_batch)


class FitterInterrupt(BaseException):
    """Fit process is interrupt by one of the fitters.
    """

    def __init__(self, *args, **kwargs):
        pass


class Fitter(object):
    """Fitter
    """

    def __init__(self,
                 interval=1,
                 count=1):
        self._interval = interval
        self._count = count

    def fit(self, i, max_loop, context):
        if i % self._interval == 0:
            for _ in range(self._count):
                self._fit(i, max_loop, context)

    def _fit(self, i, max_loop, context):
        raise NotImplementedError()


class DataFitter(Fitter):
    """Data fitter
    """

    def __init__(self,
                 data_source,
                 batch_size,
                 trainer,
                 slot_name,
                 interval=1,
                 count=1):
        super(DataFitter, self).__init__(interval, count)
        if not isinstance(data_source, data.DataSource):
            raise ValueError('data_source should be an instance of training.DataSource.')
        self._ds = data_source
        if batch_size < 0:
            raise ValueError('batch_size should not be negative.')
        self._batch_size = batch_size
        if not isinstance(trainer, Trainer):
            raise ValueError('trainer should be an instance of training.Trainer.')
        self._trainable = trainer
        self._slot_name = slot_name
        self._slot = trainer.get_slot(slot_name)

    def _fit(self, i, max_loop, context):
        data_batch = self._ds.next_batch(self._batch_size)
        ret = self._slot(*data_batch)
        context[self._slot_name] = ret


class Validator(DataFitter):
    """Validator
    """

    def __init__(self,
                 data_source,
                 batch_size,
                 trainer,
                 slot_name,
                 interval=1,
                 count=1):
        super(Validator, self).__init__(
            data_source=data_source,
            batch_size=batch_size,
            trainer=trainer,
            slot_name=slot_name,
            interval=interval,
            count=count
        )

    def _fit(self, i, max_loop, context):
        ret_list = []
        ret_dict = collections.defaultdict(float)
        if hasattr(self._ds, 'next_batch_one_pass'):
            next_batch = self._ds.__getattribute__('next_batch_one_pass')
            size = 0
            while True:
                data_batch = next_batch(self._batch_size)
                if data_batch is None:
                    break
                size += len(data_batch[0])
                ret = self._slot(*data_batch)
                if isinstance(ret, (tuple, list)):
                    ret_list.append(ret)
                elif isinstance(ret, (dict, collections.OrderedDict)):
                    for name, value in ret.items():
                        ret_dict[name] += value
                else:
                    # Should not be reached, since Slot ALWAYS returns tuple or dict.
                    raise RuntimeError('Invalid Slot outputs type.')
        else:
            data_batch = self._ds.next_batch(0)
            size = len(data_batch[0])
            batch_size = self._batch_size
            for i in range(1, size // batch_size + 1):
                data_batch = tuple(comp[(i - 1) * batch_size: i * batch_size] for comp in data_batch)
                ret = self._slot(*data_batch)
                if isinstance(ret, (tuple, list)):
                    ret_list.append(ret)
                elif isinstance(ret, (dict, collections.OrderedDict)):
                    for name, value in ret.items():
                        ret_dict[name] += value
                else:
                    # Should not be reached, since Slot ALWAYS returns tuple or dict.
                    raise RuntimeError('Invalid Slot outputs type.')
            last_size = size % batch_size
            if last_size != 0:
                data_batch = tuple(comp[-last_size:] for comp in data_batch)
                ret = self._slot(*data_batch)
                if isinstance(ret, (tuple, list)):
                    ret_list.append(ret)
                elif isinstance(ret, (dict, collections.OrderedDict)):
                    for name, value in ret.items():
                        ret_dict[name] += value
                else:
                    # Should not be reached, since Slot ALWAYS returns tuple or dict.
                    raise RuntimeError('Invalid Slot outputs type.')
        if len(ret_list) != 0:
            context[self._slot_name] = tuple(comp for comp in np.sum(ret_list, axis=0) / size)
        else:
            context[self._slot_name] = {name: value / size for name, value in ret_dict.items()}


class ScreenLogger(Fitter):
    """Screen logger
    """

    def __init__(self,
                 context,
                 value_names=('loss',),
                 message=None,
                 interval=1,
                 count=1):
        super(ScreenLogger, self).__init__(interval, count)
        self._context = context
        self._value_names = value_names
        self._message = message

    def _fit(self, i, max_loop, context):
        now = dt.datetime.now()
        print(now.strftime('[%Y-%m-%d %H:%M:%S '), end='')
        percentage = '%.2f' % (i / max_loop * 100,)
        print('%s/%s|%s%%]' % (str(i), str(max_loop), percentage), end='')
        #
        if self._message is not None:
            print('\t' + str(self._message), end='')
        #
        values = context[self._context] if self._context in context else ()
        if isinstance(values, (tuple, list)):
            for i, name in enumerate(self._value_names):
                if i < len(values):
                    value = values[i]
                    print('\t%s=%f' % (name, value), end='')
                else:
                    print('\t%s=?' % (name,), end='')
        elif isinstance(values, (dict, collections.OrderedDict)):
            for name in self._value_names:
                if name in values:
                    value = values[name]
                    print('\t%s=%f' % (name, value), end='')
                else:
                    print('\t%s=?' % (name,), end='')
        print()


class MPIDispatcher(Fitter):
    """MPI Dispatcher

    This class is used for the distributional training of the model. (Based on MPI).
    So, the servers should have one of the MPI implementation (e.g., openmpi, mpich) installed.
    If this fitter is instanced and added to a trainer, the program should be run using the MPI command:

        mpiexec -n {num_processes} python3 {python_file.py}
    """

    def __init__(self,
                 sync_interval=2):
        super(MPIDispatcher, self).__init__(1, 1)
        from mpi4py import MPI
        self._sync_interval = sync_interval
        #
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        #
        # This is very important since we should let the processes to use DIFFERENT GPUs of the same server.
        # While, if the processes run on different servers, this can cause problems.
        # TODO: Thus we need to further modify the assign policy to choose the GPU automatically.
        gpu_list = [int(item) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpu = gpu_list[self._rank % len(gpu_list)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    def _fit(self, i, max_loop, context):
        trainer = context[settings.CONTEXT_TRAINER]
        if i == 1:
            self._init_all(trainer)
        elif i % self._sync_interval == 0:
            self._update_all(trainer)

    def _init_all(self, trainer):
        if self._rank == 0:
            self._comm.bcast(trainer.parameters, root=0)
        else:
            trainer.parameters = self._comm.bcast(None, root=0)

    def _update_all(self, trainer):
        if self._rank == 0:
            #
            # Gather parameters from all processes (include the master itself).
            # Compute the mean value for each parameter.
            # Then, broadcast them.
            param_list = self._comm.gather(trainer.parameters, root=0)
            new_params = collections.defaultdict(list)
            for params in param_list:
                for name, value in params.items():
                    new_params[name].append(value)
            new_params = {key: np.mean(value_list, axis=0) for key, value_list in new_params.items()}
            new_params = trainer.parameters = self._comm.bcast(new_params, root=0)
        else:
            self._comm.gather(trainer.parameters, root=0)
            new_params = self._comm.bcast(None, root=0)
        #
        # Update the parameters to the same version for all processes.
        trainer.parameters = new_params


class OptimizerWrapper(object):
    """OptimizerWrapper
    """

    def __init__(self,
                 optimizer):
        self._optimizer = optimizer

    @property
    def optimizer(self):
        return self._optimizer

    def minimize(self, loss, var_list=None):
        pair_list = self._optimizer.compute_gradients(loss, var_list=var_list)
        pair_list = self._process_gradients(pair_list)
        return self._optimizer.apply_gradients(pair_list)

    def _process_gradients(self, pair_list):
        raise NotImplementedError


class GradientClipping(OptimizerWrapper):
    """GradientClipping
    """

    def __init__(self, optimizer, max_norm):
        self._max_norm = max_norm
        super(GradientClipping, self).__init__(optimizer)

    @property
    def max_norm(self):
        return self._max_norm

    def _process_gradients(self, pair_list):
        pair_list, raw_grad, grad = operations.clip_gradient(pair_list, self._max_norm)
        self._raw_grad_norm = raw_grad
        self._grad_norm = grad
        return pair_list

    @property
    def raw_grad_norm(self):
        return self._raw_grad_norm

    @property
    def grad_norm(self):
        return self._grad_norm
