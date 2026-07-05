""" Test TorchAdapter Model
"""
import copy
import importlib
import os
import sys
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal, assert_array_equal

from libact.base.dataset import Dataset
from libact.base.interfaces import ContinuousModel, ProbabilisticModel

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False


def make_blobs(n_per_class=20, n_features=4, labels=(0, 1, 2),
               random_state=0):
    """Well-separated gaussian blobs, one per label."""
    rng = np.random.RandomState(random_state)
    X, y = [], []
    for i, label in enumerate(labels):
        center = np.zeros(n_features)
        center[i % n_features] = 5.0
        X.append(rng.randn(n_per_class, n_features) + center)
        y += [label] * n_per_class
    return np.vstack(X), np.array(y)


if HAS_TORCH:
    def make_net(n_features=4, n_classes=3, hidden=16):
        return nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    class RiggedNet(nn.Module):
        """Constant-output net: forward ignores the input, gradients are
        zero, so training never changes its (recognizable) output."""

        def __init__(self, outputs):
            super().__init__()
            self.outputs = torch.tensor(outputs, dtype=torch.float32)
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return (self.outputs.unsqueeze(0).expand(x.shape[0], -1)
                    + 0.0 * self.dummy)


class ImportGuardTestCase(unittest.TestCase):

    def test_models_package_imports_without_torch(self):
        # Freshly (re-)import the adapter with torch blocked from
        # sys.modules; libact.models must import fine and TorchAdapter must
        # raise a helpful ImportError at construction time.
        names = ['libact.models', 'libact.models.torch_adapter']
        saved = {n: sys.modules.pop(n, None) for n in names}
        parent = sys.modules['libact']
        try:
            with mock.patch.dict(sys.modules, {'torch': None}):
                models = importlib.import_module('libact.models')
                self.assertTrue(hasattr(models, 'TorchAdapter'))
                ta_mod = sys.modules['libact.models.torch_adapter']
                self.assertIsNone(ta_mod.torch)
                with self.assertRaises(ImportError) as cm:
                    models.TorchAdapter(object(), [0, 1])
                self.assertIn("pip install libact[torch]", str(cm.exception))
                self.assertIsInstance(cm.exception.__cause__, ImportError)
        finally:
            for n in names:
                sys.modules.pop(n, None)
                if saved[n] is not None:
                    sys.modules[n] = saved[n]
            if saved['libact.models'] is not None:
                parent.models = saved['libact.models']
                sys.modules['libact.models'].torch_adapter = \
                    saved['libact.models.torch_adapter']

    def test_pyproject_declares_torch_extra(self):
        here = os.path.dirname(os.path.abspath(__file__))
        pyproject = os.path.normpath(
            os.path.join(here, '..', '..', '..', 'pyproject.toml'))
        if not os.path.exists(pyproject):
            self.skipTest("pyproject.toml not found (installed package)")
        with open(pyproject) as f:
            content = f.read()
        self.assertIn('torch = ["torch>=2.0"]', content)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class ConstructorValidationTestCase(unittest.TestCase):

    def test_module_wrong_type(self):
        from libact.models import TorchAdapter
        with self.assertRaises(TypeError):
            TorchAdapter(42, [0, 1])

    def test_optimizer_instance_rejected(self):
        from libact.models import TorchAdapter
        net = make_net()
        opt_instance = torch.optim.SGD(net.parameters(), lr=0.1)
        with self.assertRaises(TypeError):
            TorchAdapter(net, [0, 1, 2], optimizer=opt_instance)

    def test_optimizer_non_optimizer_class_rejected(self):
        from libact.models import TorchAdapter
        with self.assertRaises(TypeError):
            TorchAdapter(make_net(), [0, 1, 2], optimizer=int)
        with self.assertRaises(TypeError):
            TorchAdapter(make_net(), [0, 1, 2], optimizer='adam')

    def test_classes_too_few(self):
        from libact.models import TorchAdapter
        with self.assertRaises(ValueError):
            TorchAdapter(make_net(), [0])

    def test_classes_duplicates(self):
        from libact.models import TorchAdapter
        with self.assertRaises(ValueError):
            TorchAdapter(make_net(), [0, 1, 1])

    def test_lr_in_optimizer_kwargs(self):
        from libact.models import TorchAdapter
        with self.assertRaises(ValueError):
            TorchAdapter(make_net(), [0, 1, 2], optimizer_kwargs={'lr': 0.1})

    def test_scalar_ranges(self):
        from libact.models import TorchAdapter
        with self.assertRaises(ValueError):
            TorchAdapter(make_net(), [0, 1, 2], lr=0.0)
        with self.assertRaises(ValueError):
            TorchAdapter(make_net(), [0, 1, 2], n_epochs=0)
        with self.assertRaises(ValueError):
            TorchAdapter(make_net(), [0, 1, 2], batch_size=0)

    def test_output_invalid(self):
        from libact.models import TorchAdapter
        with self.assertRaises(ValueError):
            TorchAdapter(make_net(), [0, 1, 2], output='banana')

    def test_output_probs_requires_criterion(self):
        from libact.models import TorchAdapter
        with self.assertRaises(ValueError):
            TorchAdapter(make_net(), [0, 1, 2], output='probs')

    def test_invalid_device(self):
        from libact.models import TorchAdapter
        with self.assertRaises(RuntimeError):
            TorchAdapter(make_net(), [0, 1, 2], device='not-a-device')

    def test_invalid_random_state(self):
        from libact.models import TorchAdapter
        with self.assertRaises(ValueError):
            TorchAdapter(make_net(), [0, 1, 2], random_state='not-a-seed')


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class TrainPredictTestCase(unittest.TestCase):

    def setUp(self):
        from libact.models import TorchAdapter
        self.TorchAdapter = TorchAdapter
        self.X, self.y = make_blobs()
        self.dataset = Dataset(self.X, self.y)

    def make_adapter(self, **kwargs):
        kwargs.setdefault('random_state', 1126)
        return self.TorchAdapter(make_net(), [0, 1, 2], **kwargs)

    def test_is_probabilistic_model(self):
        model = self.make_adapter()
        self.assertIsInstance(model, ProbabilisticModel)
        self.assertIsInstance(model, ContinuousModel)

    def test_train_returns_self(self):
        model = self.make_adapter()
        self.assertIs(model.train(self.dataset), model)

    def test_predict_shape_and_labels(self):
        model = self.make_adapter().train(self.dataset)
        pred = model.predict(self.X)
        self.assertEqual(pred.shape, (len(self.X),))
        self.assertTrue(set(pred.tolist()) <= {0, 1, 2})
        self.assertEqual(pred.dtype, model.classes_.dtype)

    def test_predict_proba_shape_and_rows(self):
        model = self.make_adapter().train(self.dataset)
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.X), 3))
        self.assertEqual(proba.dtype, np.float64)
        assert_array_almost_equal(proba.sum(axis=1), np.ones(len(self.X)))

    def test_predict_real_equals_predict_proba(self):
        model = self.make_adapter().train(self.dataset)
        assert_array_equal(model.predict_real(self.X),
                           model.predict_proba(self.X))

    def test_score(self):
        model = self.make_adapter().train(self.dataset)
        expected = np.mean(model.predict(self.X) == self.y)
        self.assertEqual(model.score(self.dataset), expected)

    def test_learns_separable_blobs(self):
        model = self.make_adapter(n_epochs=50).train(self.dataset)
        self.assertGreater(model.score(self.dataset), 0.8)

    def test_untrained_raises(self):
        for module in (make_net(), make_net):  # instance and factory mode
            model = self.TorchAdapter(module, [0, 1, 2])
            for call in (lambda: model.predict(self.X),
                         lambda: model.predict_proba(self.X),
                         lambda: model.predict_real(self.X),
                         lambda: model.score(self.dataset)):
                with self.assertRaises(RuntimeError):
                    call()

    def test_train_empty_dataset(self):
        model = self.make_adapter()
        empty = Dataset(self.X, [None] * len(self.X))
        with self.assertRaises(ValueError):
            model.train(empty)

    def test_unseen_label_raises_before_training(self):
        net = make_net()
        model = self.TorchAdapter(net, [0, 1, 2])
        before = copy.deepcopy(net.state_dict())
        bad = Dataset(self.X, np.where(self.y == 2, 5, self.y))
        with self.assertRaises(ValueError) as cm:
            model.train(bad)
        self.assertIn('5', str(cm.exception))
        self.assertIn('[0, 1, 2]', str(cm.exception))
        after = net.state_dict()
        for key in before:  # weights untouched: no training step ran
            self.assertTrue(torch.equal(before[key], after[key]))

    def test_output_width_mismatch(self):
        model = self.TorchAdapter(make_net(n_classes=4), [0, 1, 2])
        with self.assertRaises(ValueError) as cm:
            model.train(self.dataset)
        self.assertIn('4', str(cm.exception))
        self.assertIn('3', str(cm.exception))

    def test_one_dimensional_output_raises(self):
        # A squeezed (n,) output makes shape[-1] equal the batch size; with a
        # batch that covers all 3 samples it would coincidentally equal the 3
        # declared classes, so only an ndim check catches it before the loss.
        class SqueezeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 1)

            def forward(self, x):
                return self.lin(x).reshape(-1)

        X = self.X[:3]
        y = np.array([0, 1, 2])
        model = self.TorchAdapter(SqueezeNet(), [0, 1, 2], batch_size=3)
        with self.assertRaises(ValueError) as cm:
            model.train(Dataset(X, y))
        self.assertIn('shape', str(cm.exception))

    def test_failed_train_after_success_invalidates_model(self):
        # If train() fails mid-fit after a prior success, the model must not
        # keep serving predictions from the half-reinitialized network.
        class FailAfterFirstTrain(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)
                self.fail = False

            def forward(self, x):
                if self.fail:
                    raise RuntimeError("boom")
                return self.lin(x)

        net = FailAfterFirstTrain()
        model = self.TorchAdapter(net, [0, 1, 2], random_state=1126)
        model.train(self.dataset)
        model.predict(self.X)  # works while trained

        net.fail = True
        with self.assertRaises(RuntimeError):
            model.train(self.dataset)
        # _trained lowered on failure: predict now refuses instead of
        # returning garbage from the reset weights.
        with self.assertRaises(RuntimeError) as cm:
            model.predict(self.X)
        self.assertIn('has not been trained', str(cm.exception))


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class LabelMappingTestCase(unittest.TestCase):

    def setUp(self):
        from libact.models import TorchAdapter
        self.TorchAdapter = TorchAdapter
        self.X, self.y = make_blobs(labels=(3, 7, 9))

    def test_non_contiguous_labels(self):
        model = self.TorchAdapter(make_net(), [3, 7, 9], random_state=1126,
                                  n_epochs=50)
        model.train(Dataset(self.X, self.y))
        pred = model.predict(self.X)
        self.assertTrue(set(pred.tolist()) <= {3, 7, 9})
        self.assertEqual(model.predict_proba(self.X).shape,
                         (len(self.X), 3))
        self.assertGreater(model.score(Dataset(self.X, self.y)), 0.8)

    def test_partial_pool_keeps_full_width(self):
        # Round-1 AL pools typically miss classes; proba width must not
        # depend on the labels present in the pool.
        mask = self.y != 9
        model = self.TorchAdapter(make_net(), [3, 7, 9], random_state=1126)
        model.train(Dataset(self.X[mask], self.y[mask]))
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape, (len(self.X), 3))

    def test_classes_order_preserved_not_sorted(self):
        # Rigged net always outputs argmax at column 0; with
        # classes=[9, 3, 7], predict must return 9 and column 0 of
        # predict_proba must carry the largest probability.
        net = RiggedNet([5.0, 1.0, -3.0])
        model = self.TorchAdapter(net, [9, 3, 7], random_state=1126)
        model.train(Dataset(self.X, self.y))
        pred = model.predict(self.X)
        assert_array_equal(pred, np.full(len(self.X), 9))
        proba = model.predict_proba(self.X)
        assert_array_equal(proba.argmax(axis=1), np.zeros(len(self.X)))
        expected = torch.softmax(
            torch.tensor([5.0, 1.0, -3.0]), dim=0).numpy()
        assert_array_almost_equal(proba[0], expected)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class ColdStartTestCase(unittest.TestCase):

    def setUp(self):
        from libact.models import TorchAdapter
        self.TorchAdapter = TorchAdapter
        self.X, self.y = make_blobs()
        self.ds_a = Dataset(self.X[::2], self.y[::2])
        self.ds_b = Dataset(self.X[1::2], self.y[1::2])

    def make_bn_net(self):
        return nn.Sequential(
            nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 3),
        )

    def test_same_seed_fresh_adapters_bitwise_identical(self):
        net = self.make_bn_net()
        m1 = self.TorchAdapter(copy.deepcopy(net), [0, 1, 2],
                               random_state=1126)
        m2 = self.TorchAdapter(copy.deepcopy(net), [0, 1, 2],
                               random_state=1126)
        p1 = m1.train(self.ds_a).predict_proba(self.X)
        p2 = m2.train(self.ds_a).predict_proba(self.X)
        assert_array_equal(p1, p2)

    def test_cold_start_forgets_previous_pool(self):
        # After a cold restart the previously-trained pool must not leak:
        # train(B) then train(A) equals train(A) then train(A) for equal
        # seed sequences (uses BatchNorm so buffer restore is exercised).
        net = self.make_bn_net()
        m1 = self.TorchAdapter(copy.deepcopy(net), [0, 1, 2],
                               random_state=1126)
        m2 = self.TorchAdapter(copy.deepcopy(net), [0, 1, 2],
                               random_state=1126)
        m1.train(self.ds_b)
        m1.train(self.ds_a)
        m2.train(self.ds_a)
        m2.train(self.ds_a)
        assert_array_equal(m1.predict_proba(self.X),
                           m2.predict_proba(self.X))

    def test_initial_snapshot_contains_buffers(self):
        net = self.make_bn_net()
        model = self.TorchAdapter(net, [0, 1, 2])
        buffer_keys = [k for k in model._initial_state
                       if 'running_mean' in k]
        self.assertTrue(buffer_keys)

    def test_warm_start_keeps_weights(self):
        # RiggedNet's gradients are zero, so any weight change we poke in
        # manually survives training iff re-initialization was skipped.
        for warm_start, poke_survives in ((True, True), (False, False)):
            net = RiggedNet([1.0, 0.0, -1.0])
            model = self.TorchAdapter(net, [0, 1, 2],
                                      warm_start=warm_start)
            model.train(self.ds_a)
            with torch.no_grad():
                model.module_.dummy.add_(1.0)
            model.train(self.ds_a)
            poked = bool((model.module_.dummy == 1.0).item())
            self.assertEqual(poked, poke_survives)

    def test_fresh_optimizer_every_train(self):
        instantiations = []

        class SpyAdam(torch.optim.Adam):
            def __init__(self, *args, **kwargs):
                instantiations.append(1)
                super().__init__(*args, **kwargs)

        model = self.TorchAdapter(make_net(), [0, 1, 2], optimizer=SpyAdam,
                                  warm_start=True)
        model.train(self.ds_a)
        model.train(self.ds_a)
        self.assertEqual(len(instantiations), 2)

    def test_factory_called_once_per_cold_train(self):
        calls = []

        def factory():
            calls.append(1)
            return make_net()

        model = self.TorchAdapter(factory, [0, 1, 2], random_state=1126)
        model.train(self.ds_a)
        model.train(self.ds_a)
        self.assertEqual(len(calls), 2)

    def test_factory_called_once_with_warm_start(self):
        calls = []

        def factory():
            calls.append(1)
            return make_net()

        model = self.TorchAdapter(factory, [0, 1, 2], warm_start=True)
        model.train(self.ds_a)
        model.train(self.ds_a)
        self.assertEqual(len(calls), 1)

    def test_factory_returning_non_module(self):
        model = self.TorchAdapter(lambda: 42, [0, 1, 2])
        with self.assertRaises(TypeError):
            model.train(self.ds_a)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class InputHandlingTestCase(unittest.TestCase):

    def setUp(self):
        from libact.models import TorchAdapter
        self.TorchAdapter = TorchAdapter
        self.X, self.y = make_blobs()

    def test_sparse_dataset_matches_dense(self):
        net = make_net()
        dense = self.TorchAdapter(copy.deepcopy(net), [0, 1, 2],
                                  random_state=1126)
        sparse = self.TorchAdapter(copy.deepcopy(net), [0, 1, 2],
                                   random_state=1126)
        p_dense = dense.train(
            Dataset(self.X, self.y)).predict_proba(self.X)
        p_sparse = sparse.train(
            Dataset(sp.csr_matrix(self.X), self.y)).predict_proba(self.X)
        assert_array_almost_equal(p_dense, p_sparse)

    def test_sparse_feature_at_inference(self):
        model = self.TorchAdapter(make_net(), [0, 1, 2], random_state=1126)
        model.train(Dataset(self.X, self.y))
        assert_array_almost_equal(
            model.predict_proba(sp.csr_matrix(self.X)),
            model.predict_proba(self.X))
        assert_array_equal(model.predict(sp.csr_matrix(self.X)),
                           model.predict(self.X))

    def test_non_sliceable_sparse_formats_at_inference(self):
        # coo/bsr do not support row slicing; the adapter must convert to a
        # sliceable format instead of crashing in the inference batch loop.
        model = self.TorchAdapter(make_net(), [0, 1, 2], random_state=1126)
        model.train(Dataset(self.X, self.y))
        dense = model.predict_proba(self.X)
        csr = sp.csr_matrix(self.X)
        for fmt in ('coo', 'csc', 'lil', 'bsr'):
            proba = model.predict_proba(csr.asformat(fmt))
            assert_array_almost_equal(proba, dense,
                                      err_msg='format %s' % fmt)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class InferenceSemanticsTestCase(unittest.TestCase):

    def setUp(self):
        from libact.models import TorchAdapter
        self.TorchAdapter = TorchAdapter
        self.X, self.y = make_blobs()
        self.dataset = Dataset(self.X, self.y)

    def make_dropout_net(self):
        return nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 3),
        )

    def test_eval_mode_predictions_are_deterministic(self):
        model = self.TorchAdapter(self.make_dropout_net(), [0, 1, 2],
                                  random_state=1126)
        model.train(self.dataset)
        assert_array_equal(model.predict_proba(self.X),
                           model.predict_proba(self.X))

    def test_module_mode_toggling(self):
        model = self.TorchAdapter(make_net(), [0, 1, 2], random_state=1126)
        model.train(self.dataset)
        self.assertTrue(model.module_.training)  # left in train mode
        model.predict(self.X)
        self.assertFalse(model.module_.training)  # eval engaged

    def test_batched_inference_matches_single_batch(self):
        model = self.TorchAdapter(make_net(), [0, 1, 2], random_state=1126)
        model.train(self.dataset)
        model.batch_size = len(self.X)
        whole = model.predict_proba(self.X)
        model.batch_size = 7  # does not divide 60 evenly
        batched = model.predict_proba(self.X)
        assert_array_almost_equal(whole, batched)

    def test_output_probs_no_double_softmax(self):
        probs = [0.7, 0.2, 0.1]
        net = RiggedNet(probs)
        model = self.TorchAdapter(net, [0, 1, 2], output='probs',
                                  criterion=nn.NLLLoss())
        model.train(self.dataset)
        proba = model.predict_proba(self.X)
        assert_array_almost_equal(proba, np.tile(probs, (len(self.X), 1)))


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class DeterminismTestCase(unittest.TestCase):

    def setUp(self):
        from libact.models import TorchAdapter
        self.TorchAdapter = TorchAdapter
        self.X, self.y = make_blobs()
        self.dataset = Dataset(self.X, self.y)

    @staticmethod
    def dropout_factory():
        return nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 3),
        )

    def test_seeded_runs_bitwise_identical(self):
        # Factory mode + dropout: pins that weight init AND dropout masks
        # are seeded (whole-train-body fork_rng).
        p = []
        for _ in range(2):
            model = self.TorchAdapter(self.dropout_factory, [0, 1, 2],
                                      random_state=1126)
            p.append(model.train(self.dataset).predict_proba(self.X))
        assert_array_equal(p[0], p[1])

    def test_consecutive_trains_draw_different_round_seeds(self):
        model = self.TorchAdapter(self.dropout_factory, [0, 1, 2],
                                  random_state=1126, n_epochs=2)
        p1 = model.train(self.dataset).predict_proba(self.X)
        p2 = model.train(self.dataset).predict_proba(self.X)
        self.assertFalse(np.array_equal(p1, p2))

    def test_seeded_train_does_not_perturb_global_torch_rng(self):
        torch.manual_seed(0)
        r1 = torch.rand(3)
        torch.manual_seed(0)
        model = self.TorchAdapter(self.dropout_factory, [0, 1, 2],
                                  random_state=1126)
        model.train(self.dataset)
        r2 = torch.rand(3)
        self.assertTrue(torch.equal(r1, r2))

    def test_unseeded_train_consumes_global_torch_rng(self):
        torch.manual_seed(0)
        r1 = torch.rand(3)
        torch.manual_seed(0)
        model = self.TorchAdapter(self.dropout_factory, [0, 1, 2])
        model.train(self.dataset)
        r2 = torch.rand(3)
        self.assertFalse(torch.equal(r1, r2))

    @unittest.skipUnless(HAS_CUDA, 'CUDA is not available')
    def test_seeded_cpu_train_does_not_perturb_cuda_rng(self):
        # torch.manual_seed inside the fork seeds CUDA generators too; the fork
        # must restore them so a seeded CPU adapter does not disturb the CUDA
        # RNG stream of surrounding GPU code.
        torch.cuda.manual_seed_all(0)
        r1 = torch.rand(3, device='cuda')
        torch.cuda.manual_seed_all(0)
        model = self.TorchAdapter(self.dropout_factory, [0, 1, 2],
                                  device='cpu', random_state=1126)
        model.train(self.dataset)
        r2 = torch.rand(3, device='cuda')
        self.assertTrue(torch.equal(r1, r2))

    def test_seeded_al_runs_identical(self):
        from libact.labelers import IdealLabeler
        from libact.query_strategies import UncertaintySampling

        def run_once():
            X, y = make_blobs(n_per_class=15, random_state=7)
            n_labeled = 9
            # first 9 samples cover all 3 classes (interleave)
            order = np.argsort(np.tile(np.arange(15), 3), kind='stable')
            X, y = X[order], y[order]
            labels = np.concatenate(
                [y[:n_labeled], [None] * (len(y) - n_labeled)])
            trn_ds = Dataset(X, labels)
            labeler = IdealLabeler(Dataset(X, y))
            model = self.TorchAdapter(self.dropout_factory, [0, 1, 2],
                                      random_state=1126)
            qs = UncertaintySampling(trn_ds, method='lc', model=model)
            queried = []
            for _ in range(5):
                ask_id = qs.make_query()
                queried.append(ask_id)
                trn_ds.update(ask_id, labeler.label(X[ask_id]))
            return queried, model.predict_proba(X)

        q1, p1 = run_once()
        q2, p2 = run_once()
        self.assertEqual(q1, q2)
        assert_array_equal(p1, p2)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class CloneTestCase(unittest.TestCase):

    def setUp(self):
        from libact.models import TorchAdapter
        self.TorchAdapter = TorchAdapter
        self.X, self.y = make_blobs()
        self.dataset = Dataset(self.X, self.y)

    def test_clone_does_not_share_module(self):
        model = self.TorchAdapter(make_net(), [0, 1, 2], random_state=1126)
        cloned = model.clone()
        self.assertIsNot(cloned._module_arg, model._module_arg)

    def test_training_clone_leaves_original_untouched(self):
        model = self.TorchAdapter(make_net(), [0, 1, 2], random_state=1126)
        model.train(self.dataset)
        before = model.predict_proba(self.X)
        cloned = model.clone()
        cloned.train(Dataset(self.X[::2], self.y[::2]))
        assert_array_equal(model.predict_proba(self.X), before)

    def test_clone_of_trained_adapter_is_unfitted(self):
        model = self.TorchAdapter(make_net(), [0, 1, 2], random_state=1126)
        model.train(self.dataset)
        cloned = model.clone()
        with self.assertRaises(RuntimeError):
            cloned.predict(self.X)

    def test_clone_seed_derivation(self):
        seeded = self.TorchAdapter(make_net(), [0, 1, 2], random_state=1126)
        self.assertTrue(seeded.clone()._seeded)
        unseeded = self.TorchAdapter(make_net(), [0, 1, 2])
        self.assertFalse(unseeded.clone()._seeded)

    def test_clone_preserves_config(self):
        model = self.TorchAdapter(
            make_net(), [0, 1, 2], optimizer=torch.optim.SGD,
            optimizer_kwargs={'momentum': 0.9}, lr=0.05, n_epochs=3,
            batch_size=8, warm_start=True, random_state=1126)
        cloned = model.clone()
        self.assertIs(cloned.optimizer, torch.optim.SGD)
        self.assertEqual(cloned.optimizer_kwargs, {'momentum': 0.9})
        self.assertEqual(cloned.lr, 0.05)
        self.assertEqual(cloned.n_epochs, 3)
        self.assertEqual(cloned.batch_size, 8)
        self.assertTrue(cloned.warm_start)
        assert_array_equal(cloned.classes_, model.classes_)

    def test_bald_integration(self):
        from libact.query_strategies import BALD
        X, y = make_blobs(n_per_class=15)
        # interleave so all classes appear among the first 20 labeled
        order = np.argsort(np.tile(np.arange(15), 3), kind='stable')
        X, y = X[order], y[order]
        dataset = Dataset(X, np.concatenate(
            [y[:20], [None] * (len(y) - 20)]))
        base = self.TorchAdapter(make_net(), [0, 1, 2], random_state=1126,
                                 n_epochs=5)
        qs = BALD(dataset, base_model=base, n_models=3, random_state=1)
        ask_id = qs.make_query()
        unlabeled_ids, _ = dataset.get_unlabeled_entries()
        self.assertIn(ask_id, unlabeled_ids)
        probas = [m.predict_proba(X) for m in qs.models]
        self.assertFalse(all(np.array_equal(probas[0], p)
                             for p in probas[1:]))


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class DeviceTestCase(unittest.TestCase):

    def setUp(self):
        from libact.models import TorchAdapter
        self.TorchAdapter = TorchAdapter
        self.X, self.y = make_blobs()
        self.dataset = Dataset(self.X, self.y)

    def test_device_accepts_str_and_torch_device(self):
        for device in ('cpu', torch.device('cpu')):
            model = self.TorchAdapter(make_net(), [0, 1, 2], device=device,
                                      random_state=1126)
            model.train(self.dataset)
            param_device = next(model.module_.parameters()).device
            self.assertEqual(param_device.type, 'cpu')
            proba = model.predict_proba(self.X)
            self.assertIsInstance(proba, np.ndarray)

    @unittest.skipUnless(HAS_CUDA, 'CUDA is not available')
    def test_cuda_device_trains_and_predicts(self):
        model = self.TorchAdapter(make_net(), [0, 1, 2], device='cuda',
                                  random_state=1126, n_epochs=40)
        model.train(self.dataset)
        param_device = next(model.module_.parameters()).device
        self.assertEqual(param_device.type, 'cuda')
        proba = model.predict_proba(self.X)  # returned on CPU as numpy
        self.assertIsInstance(proba, np.ndarray)
        self.assertEqual(proba.shape, (len(self.X), 3))
        self.assertGreater(model.score(self.dataset), 0.8)


if __name__ == '__main__':
    unittest.main()
