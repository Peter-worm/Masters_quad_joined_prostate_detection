import unittest

from src.utils.cross_validation import create_folds, split_train_validation


class CrossValidationTests(unittest.TestCase):
    def test_folds_are_reproducible_disjoint_and_complete(self):
        patients = [f"patient-{index}" for index in range(20)]

        folds = create_folds(patients, n_splits=5, random_seed=42)
        repeated_folds = create_folds(patients, n_splits=5, random_seed=42)

        self.assertEqual(folds, repeated_folds)
        self.assertEqual(len(folds), 5)
        self.assertEqual(set().union(*map(set, folds)), set(patients))
        self.assertEqual(sum(len(fold) for fold in folds), len(patients))
        for index, fold in enumerate(folds):
            for other_fold in folds[index + 1 :]:
                self.assertTrue(set(fold).isdisjoint(other_fold))

    def test_requires_at_least_two_folds(self):
        with self.assertRaises(ValueError):
            create_folds(["patient-1"], n_splits=1)

    def test_creates_requested_number_of_folds_when_sizes_are_uneven(self):
        folds = create_folds([f"patient-{index}" for index in range(6)], n_splits=5)

        self.assertEqual(len(folds), 5)
        self.assertEqual(sorted(len(fold) for fold in folds), [1, 1, 1, 1, 2])

    def test_train_validation_split_is_reproducible_and_complete(self):
        items = [f"patient-{index}" for index in range(10)]

        train, validation = split_train_validation(items, validation_fraction=0.2)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(validation), 2)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertEqual(set(train) | set(validation), set(items))


if __name__ == "__main__":
    unittest.main()
