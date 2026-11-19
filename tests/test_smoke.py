"""Small tests for a student demo; run with unittest discover -s tests -v."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import FEATURES, check_features, check_labels, make_pipeline, predict_frame
from generate_data import generate


class DemoModelTests(unittest.TestCase):
    def test_generated_data_shape_and_labels(self):
        df = generate(30, 17)
        self.assertEqual(len(df), 60)
        self.assertEqual(set(df["label"]), {0, 1})
        self.assertEqual(len(FEATURES), 9)

    def test_missing_field_is_rejected(self):
        df = generate(10, 42).drop(columns=["dst_port"])
        with self.assertRaisesRegex(ValueError, "Missing CSV"):
            check_features(df)

    def test_labels_require_two_classes(self):
        df = generate(20, 44)
        self.assertEqual(set(check_labels(df)), {0, 1})
        with self.assertRaises(ValueError):
            check_labels(df.assign(label=0))

    def test_model_predicts_without_crashing(self):
        df = generate(30, 101)
        model = make_pipeline()
        model.set_params(classifier__max_iter=40, classifier__early_stopping=False)
        model.fit(check_features(df), check_labels(df))
        results = predict_frame(model, df.drop(columns=["label"]).head(3))
        self.assertEqual(len(results), 3)
        self.assertTrue(results["suspicious_score"].between(0, 1).all())
        self.assertTrue(set(results["prediction"]).issubset({"Normal", "Suspicious"}))


if __name__ == "__main__":
    unittest.main()
