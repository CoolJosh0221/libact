"""Integration tests for Dataset.update_batch with the real, stateful
observers that rely on the per-entry (entry_id, label) callback: QUIRE
(index bookkeeping), QueryByCommittee (committee retrain per label),
EpsilonUncertaintySampling (model retrain per label) and
ActiveLearningByLearning (bandit bookkeeping with preconditions).
"""
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from libact.base.dataset import Dataset
from libact.models import SklearnProbaAdapter
from libact.query_strategies import (
    ActiveLearningByLearning,
    EpsilonUncertaintySampling,
    QueryByCommittee,
    QUIRE,
    UncertaintySampling,
)


def _make_dataset():
    np.random.seed(1126)
    X = np.random.randn(30, 5)
    y = np.random.choice([0, 1], size=30)
    return Dataset(X, list(y[:10]) + [None] * 20), y


class TestUpdateBatchWithQuire(unittest.TestCase):

    def test_end_state_matches_sequential(self):
        ds_batch, y = _make_dataset()
        ds_seq, _ = _make_dataset()
        quire_batch = QUIRE(ds_batch)
        quire_seq = QUIRE(ds_seq)

        ids = [12, 15, 20]
        labels = [int(y[i]) for i in ids]

        ds_batch.update_batch(ids, labels)
        for i, label in zip(ids, labels):
            ds_seq.update(i, label)

        self.assertEqual(quire_batch.Uindex, quire_seq.Uindex)
        self.assertEqual(quire_batch.Lindex, quire_seq.Lindex)
        self.assertEqual(list(quire_batch.y), list(quire_seq.y))
        # The newly labeled ids moved from Uindex to Lindex.
        for i in ids:
            self.assertIn(i, quire_batch.Lindex)
            self.assertNotIn(i, quire_batch.Uindex)


class TestUpdateBatchWithQueryByCommittee(unittest.TestCase):

    def test_committee_retrains_once_per_entry(self):
        ds, y = _make_dataset()
        np.random.seed(0)
        qbc = QueryByCommittee(
            ds,
            models=[
                SklearnProbaAdapter(
                    LogisticRegression(C=c, max_iter=200, solver='liblinear')
                )
                for c in [0.1, 1.0]
            ],
            random_state=0,
        )

        calls = []
        original = qbc.teach_students

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        qbc.teach_students = spy

        ids = [11, 14, 17]
        ds.update_batch(ids, [int(y[i]) for i in ids])
        # Exactly the same notification stream as 3 sequential updates.
        self.assertEqual(len(calls), 3)


class TestUpdateBatchWithEpsilonUS(unittest.TestCase):

    def test_observer_retrains_without_error(self):
        ds, y = _make_dataset()
        qs = EpsilonUncertaintySampling(
            ds,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            epsilon=0.1,
            random_state=0,
        )
        ids = [13, 18]
        ds.update_batch(ids, [int(y[i]) for i in ids])
        # The strategy stays usable after the bulk update.
        self.assertIsInstance(qs.make_query(), (int, np.integer))


class TestUpdateBatchWithALBL(unittest.TestCase):

    def _make_albl(self):
        ds, y = _make_dataset()
        albl = ActiveLearningByLearning(
            ds,
            query_strategies=[
                UncertaintySampling(
                    ds,
                    model=SklearnProbaAdapter(
                        LogisticRegression(
                            C=1., max_iter=200, solver='liblinear')
                    )
                ),
            ],
            T=10,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            random_state=42,
        )
        return ds, y, albl

    def test_queried_id_through_update_batch(self):
        # The supported flow: ids obtained from ALBL's own make_query may
        # be applied through update_batch, and ALBL's bookkeeping matches
        # the sequential path.
        ds_b, y_b, albl_b = self._make_albl()
        ds_s, y_s, albl_s = self._make_albl()

        ask_b = albl_b.make_query()
        ask_s = albl_s.make_query()
        # Identical fixtures and seeds: both instances pick the same id.
        self.assertEqual(ask_b, ask_s)

        ds_b.update_batch([ask_b], [int(y_b[ask_b])])
        ds_s.update(ask_s, int(y_s[ask_s]))

        self.assertEqual(albl_b.queried_hist_, albl_s.queried_hist_)
        self.assertEqual(albl_b.W, albl_s.W)

    def test_update_before_make_query_fails_like_sequential(self):
        # ALBL assumes each update corresponds to an entry it has itself
        # queried via make_query() — before any query its query
        # distribution is uninitialized and update() raises TypeError.
        # ALBL does NOT otherwise guard against unqueried ids (see the
        # parity test below); what update_batch must guarantee is that it
        # fails exactly like the sequential update() path does.
        ds_b, y_b, _albl_b = self._make_albl()
        ds_s, y_s, _albl_s = self._make_albl()
        with self.assertRaises(TypeError):
            ds_b.update_batch([15], [int(y_b[15])])
        with self.assertRaises(TypeError):
            ds_s.update(15, int(y_s[15]))

    def test_arbitrary_id_parity_with_sequential(self):
        # Pre-existing ALBL behavior: after make_query(), updating an id
        # ALBL did NOT query is accepted silently (the bandit bookkeeping
        # uses whatever query distribution is current). That behavior is
        # out of scope to change here — this test pins that update_batch
        # reproduces the sequential path exactly, corrupt bookkeeping
        # included.
        ds_b, y_b, albl_b = self._make_albl()
        ds_s, y_s, albl_s = self._make_albl()

        ask_b = albl_b.make_query()
        ask_s = albl_s.make_query()
        self.assertEqual(ask_b, ask_s)

        # Pick an unlabeled id that ALBL did not query.
        unlabeled_ids = ds_b.get_unlabeled_entries()[0].tolist()
        other = next(i for i in unlabeled_ids if i != ask_b)

        ds_b.update_batch([other], [int(y_b[other])])
        ds_s.update(other, int(y_s[other]))

        self.assertEqual(albl_b.queried_hist_, albl_s.queried_hist_)
        self.assertEqual(albl_b.W, albl_s.W)


if __name__ == '__main__':
    unittest.main()
