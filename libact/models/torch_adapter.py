"""PyTorch neural network classifier adapter

This module provides :class:`TorchAdapter`, which wraps a PyTorch
``torch.nn.Module`` so that a neural network can be used as the model in
libact's active-learning loop.

PyTorch is an *optional* dependency of libact: importing this module (and
``libact.models``) works without torch installed, but constructing a
:class:`TorchAdapter` then raises an :class:`ImportError` with installation
instructions.
"""
import copy

import numpy as np
import scipy.sparse as sp

from libact.base.interfaces import ProbabilisticModel
from libact.utils import seed_random_state

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_IMPORT_ERROR = None
except ImportError as e:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = e


class TorchAdapter(ProbabilisticModel):

    r"""Adapter for using a PyTorch ``nn.Module`` as a libact model.

    The adapter implements libact's :class:`ProbabilisticModel` interface on
    top of a user-supplied network: ``train(dataset)`` runs a mini-batch
    training loop over the labeled entries, ``predict`` returns class labels,
    and ``predict_proba`` returns class probabilities (``predict_real`` is
    inherited and equals ``predict_proba``).

    Like libact's scikit-learn based models, ``train()`` refits from scratch
    on every call by default: the network weights are re-initialized before
    each fit so that repeated training on a growing labeled pool behaves the
    same as refitting a fresh estimator. Pass ``warm_start=True`` to instead
    continue from the current weights. A fresh optimizer is constructed on
    every ``train()`` call in both modes.

    Parameters
    ----------
    module : torch.nn.Module instance or zero-argument callable
        The network to train, or a factory callable returning a fresh
        network. The module's ``forward`` must map a float32 batch of shape
        ``(n, n_features)`` to an output of shape ``(n, len(classes))``.
        When an instance is given, its initial ``state_dict`` (parameters
        and buffers) is snapshotted at construction and, on each cold-start
        ``train()`` call, restored before fitting; when a factory is given,
        a fresh network is built instead. With ``warm_start=True`` the
        re-initialization is skipped after the first call (see
        ``warm_start``).

    classes : sequence
        All label values the model can encounter, in the order matching the
        module's output neurons: output column ``j`` corresponds to
        ``classes[j]``. The order is preserved as given (it is *not*
        sorted), so a ``criterion`` with a ``weight=`` tensor follows this
        order. When mixing with scikit-learn based models in a committee
        (whose ``classes_`` are sorted), pass ``classes`` in sorted order so
        that probability columns align.

    criterion : loss instance or callable, optional (default=None)
        Loss called as ``criterion(outputs, target_indices)``. ``None``
        uses ``torch.nn.CrossEntropyLoss()``, which expects the module to
        output raw logits.

    optimizer : torch.optim.Optimizer subclass, optional (default=None)
        The optimizer *class* (not an instance -- an instance would be bound
        to parameter tensors that are replaced on re-initialization).
        ``None`` uses ``torch.optim.Adam``.

    optimizer_kwargs : dict, optional (default=None)
        Extra keyword arguments for the optimizer constructor, e.g.
        ``{'weight_decay': 1e-4}``. Must not contain ``'lr'``; use the
        ``lr`` parameter instead.

    lr : float, optional (default=1e-3)
        Learning rate, forwarded to the optimizer constructor.

    n_epochs : int, optional (default=10)
        Number of passes over the labeled pool per ``train()`` call.

    batch_size : int, optional (default=32)
        Mini-batch size for the training DataLoader and for batched
        inference.

    device : str or torch.device, optional (default='cpu')
        Device the module and batches are moved to. There is no automatic
        CUDA detection; pass ``'cuda'`` explicitly to use a GPU.

    output : {'logits', 'probs'}, optional (default='logits')
        What the module's forward pass returns. With ``'logits'``,
        ``predict_proba`` applies a softmax; with ``'probs'`` the output is
        used as-is (an explicit ``criterion`` is then required, since the
        default ``CrossEntropyLoss`` expects logits).

    warm_start : bool, optional (default=False)
        When True, ``train()`` calls after the first continue from the
        current weights instead of re-initializing them. The optimizer is
        still constructed fresh on every call (e.g. Adam moment estimates
        are never carried across calls).

    random_state : {int, np.random.RandomState instance, None}, optional
        (default=None)
        Random state of the adapter. When set, each ``train()`` call draws
        a round seed from it and runs inside ``torch.random.fork_rng`` with
        that seed, making weight initialization (factory mode), DataLoader
        shuffling, and dropout masks reproducible -- without perturbing the
        global torch RNG observed by other code. When None, torch's global
        RNG is consumed as-is.

    Attributes
    ----------
    classes\_ : numpy array, shape (n_classes,)
        The class labels, in the order given (column ``j`` of
        ``predict_proba`` is the probability of ``classes_[j]``).

    module\_ : torch.nn.Module or None
        The wrapped network (None in factory mode before the first
        ``train()`` call).

    Notes
    -----
    Sparse feature matrices are densified (the whole labeled pool at train
    time, per batch at inference), so very large sparse datasets may not fit
    in memory. The training DataLoader always shuffles and keeps the last
    incomplete batch; a trailing batch of size 1 can raise BatchNorm's own
    error during training.

    Examples
    --------
    Here is an example of using TorchAdapter to classify the iris dataset:

    .. code-block:: python

       import torch.nn as nn
       from sklearn import datasets
       from sklearn.model_selection import train_test_split

       from libact.base.dataset import Dataset
       from libact.models import TorchAdapter

       iris = datasets.load_iris()
       X_train, X_test, y_train, y_test = train_test_split(
           iris.data, iris.target, test_size=0.3)

       net = nn.Sequential(
           nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3),
       )
       model = TorchAdapter(net, classes=[0, 1, 2], n_epochs=20,
                            random_state=1126)

       model.train(Dataset(X_train, y_train))
       model.predict(X_test)
       model.predict_proba(X_test)
    """

    def __init__(self, module, classes, *, criterion=None, optimizer=None,
                 optimizer_kwargs=None, lr=1e-3, n_epochs=10, batch_size=32,
                 device='cpu', output='logits', warm_start=False,
                 random_state=None):
        if torch is None:
            raise ImportError(
                "TorchAdapter requires PyTorch, which is an optional "
                "dependency of libact. Install it with "
                "'pip install libact[torch]' or 'pip install torch'."
            ) from _TORCH_IMPORT_ERROR

        if isinstance(module, nn.Module):
            self._is_factory = False
        elif callable(module):
            self._is_factory = True
        else:
            raise TypeError(
                "module must be a torch.nn.Module instance or a "
                "zero-argument callable returning one, got %r" % type(module))
        self._module_arg = module
        self.module_ = None if self._is_factory else module

        if optimizer is None:
            optimizer = torch.optim.Adam
        elif isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer subclass (a "
                "class, not an instance), because the network is "
                "re-initialized on every train() call")
        elif not (isinstance(optimizer, type)
                  and issubclass(optimizer, torch.optim.Optimizer)):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer subclass, "
                "got %r" % (optimizer,))
        self.optimizer = optimizer

        classes = np.asarray(classes)
        if classes.ndim != 1:
            raise ValueError("classes must be a 1-d sequence of labels")
        if classes.shape[0] < 2:
            raise ValueError("classes must contain at least 2 labels")
        if np.unique(classes).shape[0] != classes.shape[0]:
            raise ValueError("classes must not contain duplicate labels")
        self.classes_ = classes
        self._class_index = {
            label: j for j, label in enumerate(classes.tolist())}

        optimizer_kwargs = dict(optimizer_kwargs) if optimizer_kwargs else {}
        if 'lr' in optimizer_kwargs:
            raise ValueError(
                "pass the learning rate via the lr parameter, not "
                "optimizer_kwargs")
        self.optimizer_kwargs = optimizer_kwargs

        if lr <= 0:
            raise ValueError("lr must be > 0")
        if n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        if output not in ('logits', 'probs'):
            raise ValueError("output must be 'logits' or 'probs'")
        if output == 'probs' and criterion is None:
            raise ValueError(
                "the default CrossEntropyLoss expects logits; supply an "
                "explicit criterion when output='probs'")
        self.output = output
        self.criterion = (criterion if criterion is not None
                          else nn.CrossEntropyLoss())

        self.device = torch.device(device)
        self.warm_start = warm_start
        self.random_state_ = seed_random_state(random_state)
        self._seeded = random_state is not None

        if not self._is_factory:
            # Snapshot parameters AND buffers (e.g. BatchNorm running
            # stats); deliberately not reset_parameters(), which silently
            # no-ops on custom modules.
            self._initial_state = copy.deepcopy(module.state_dict())

        self._trained = False

    def train(self, dataset, *args, **kwargs):
        """Train the network on the labeled entries of the given dataset.

        Parameters
        ----------
        dataset : Dataset object
            The training dataset the model is to be trained on.

        Returns
        -------
        self : TorchAdapter
        """
        X, y = dataset.format_sklearn()
        if len(y) == 0:
            raise ValueError("dataset has no labeled entries to train on")
        y_list = np.asarray(y).tolist()
        unseen = set(y_list) - set(self._class_index)
        if unseen:
            raise ValueError(
                "dataset contains label(s) %r not in classes=%r"
                % (sorted(unseen), self.classes_.tolist()))
        y_idx = np.array([self._class_index[label] for label in y_list],
                         dtype=np.int64)
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Invalidate on failure: if _fit raises after the weight snapshot has
        # been restored (and partial optimizer steps have run), the network is
        # in an inconsistent state and must not keep serving predictions as if
        # trained. _fit reads self._trained for its warm-start decision, so the
        # flag is only lowered on failure, never before _fit runs.
        try:
            if self._seeded:
                round_seed = int(self.random_state_.randint(2 ** 31 - 1))
                with torch.random.fork_rng(devices=self._fork_devices()):
                    torch.manual_seed(round_seed)
                    self._fit(X, y_idx, round_seed)
            else:
                self._fit(X, y_idx, None)
        except BaseException:
            self._trained = False
            raise

        self._trained = True
        return self

    @staticmethod
    def _fork_devices():
        # torch.manual_seed seeds the CPU generator AND every CUDA/accelerator
        # generator. torch.random.fork_rng always saves/restores the CPU
        # generator; we additionally fork all CUDA devices so the manual_seed
        # inside the fork cannot leak into CUDA RNG streams observed by other
        # code (e.g. a sibling committee member running on GPU).
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []

    def _fit(self, X, y_idx, round_seed):
        # Cold start (the default) re-initializes the network so that every
        # train() call behaves like refitting a fresh estimator.
        if self._is_factory:
            if self.module_ is None or not self.warm_start:
                built = self._module_arg()
                if not isinstance(built, nn.Module):
                    raise TypeError(
                        "module factory returned %r, expected "
                        "torch.nn.Module" % type(built))
                self.module_ = built
        else:
            if not (self.warm_start and self._trained):
                self.module_.load_state_dict(self._initial_state)

        self.module_.to(self.device)
        if isinstance(self.criterion, nn.Module):
            self.criterion.to(self.device)
        optimizer = self.optimizer(self.module_.parameters(), lr=self.lr,
                                   **self.optimizer_kwargs)

        generator = None
        if round_seed is not None:
            generator = torch.Generator().manual_seed(round_seed + 1)
        # Always shuffle: AL pools are ordered by query time, and unshuffled
        # SGD on that ordering biases the fit.
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx)),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
            num_workers=0, generator=generator)

        self.module_.train()
        width_checked = False
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                out = self.module_(xb)
                if not width_checked:
                    # Check rank too: a squeezed (n,) output would otherwise
                    # make shape[-1] equal the batch size and could pass by
                    # coincidence, then fail cryptically inside the criterion.
                    if out.ndim != 2 or out.shape[-1] != len(self.classes_):
                        raise ValueError(
                            "module must output shape (batch_size, %d) for %d "
                            "classes, but produced shape %s"
                            % (len(self.classes_), len(self.classes_),
                               tuple(out.shape)))
                    width_checked = True
                loss = self.criterion(out, yb)
                loss.backward()
                optimizer.step()

    def _forward(self, feature):
        """Batched eval-mode forward pass; returns raw outputs on CPU."""
        if not self._trained:
            raise RuntimeError(
                "TorchAdapter has not been trained yet; call train(dataset) "
                "first.")
        if sp.issparse(feature):
            # Convert to a sliceable format once: coo/dia/bsr do not support
            # row slicing, so the per-batch densification below would crash on
            # them without this. csr slicing is cheap and stays sparse until
            # the per-batch toarray().
            feature = feature.tocsr()
        else:
            feature = np.asarray(feature)
        n = feature.shape[0]
        if n == 0:
            return torch.zeros((0, len(self.classes_)))
        self.module_.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                xb = feature[start:start + self.batch_size]
                if sp.issparse(xb):
                    xb = xb.toarray()
                xb = np.asarray(xb, dtype=np.float32)
                xb = torch.from_numpy(xb).to(self.device)
                outputs.append(self.module_(xb).cpu())
        return torch.cat(outputs, dim=0)

    def predict(self, feature, *args, **kwargs):
        out = self._forward(feature)
        # argmax is monotone-invariant, so raw outputs need no softmax.
        return self.classes_[out.argmax(dim=1).numpy()]

    def predict_proba(self, feature, *args, **kwargs):
        out = self._forward(feature)
        if self.output == 'logits':
            out = torch.softmax(out, dim=1)
        return out.numpy().astype(np.float64)

    def score(self, testing_dataset, *args, **kwargs):
        X, y = testing_dataset.format_sklearn()
        return float(np.mean(self.predict(X) == y))

    def clone(self):
        """Return an unfitted TorchAdapter with the same configuration.

        In instance mode the wrapped module is deep-copied and reset to its
        initial state, so clones never share network weights (committee
        members trained separately stay separate). In factory mode the same
        factory is reused, so clones also get diverse initializations. A
        seeded adapter hands each clone a derived seed, keeping whole-run
        reproducibility.
        """
        if self._is_factory:
            module = self._module_arg
        else:
            module = copy.deepcopy(self._module_arg)
            module.load_state_dict(self._initial_state)
        if self._seeded:
            random_state = int(self.random_state_.randint(2 ** 31 - 1))
        else:
            random_state = None
        return TorchAdapter(
            module, self.classes_, criterion=self.criterion,
            optimizer=self.optimizer, optimizer_kwargs=self.optimizer_kwargs,
            lr=self.lr, n_epochs=self.n_epochs, batch_size=self.batch_size,
            device=self.device, output=self.output,
            warm_start=self.warm_start, random_state=random_state)
