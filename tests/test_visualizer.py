"""Tests for the real-model (not random) animated forward pass."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import check_features, check_labels, make_pipeline
from generate_data import generate
from neural_visualizer import inspect_forward_pass, render_animation_html


class VisualizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = generate(80, 2026)
        cls.model = make_pipeline()
        cls.model.set_params(classifier__max_iter=60, classifier__early_stopping=False)
        cls.model.fit(check_features(data), check_labels(data))
        cls.example = data.drop(columns=["label"]).iloc[[2]]

    def test_forward_pass_matches_trained_prediction(self):
        trace = inspect_forward_pass(self.model, self.example)
        self.assertEqual([len(a) for a in trace["layers"]], [11, 32, 16, 1])
        expected = self.model.predict_proba(self.example)[0, 1]
        self.assertAlmostEqual(trace["suspicious_score"], expected, places=8)
        self.assertTrue(np.all(trace["layers"][1] >= 0))
        self.assertTrue(np.all(trace["layers"][2] >= 0))

    def test_html_has_animation_controls_and_offline_rendering(self):
        result = render_animation_html(inspect_forward_pass(self.model, self.example))
        self.assertIn('id="play"', result)
        self.assertIn('id="replay"', result)
        self.assertIn('id="speed"', result)
        self.assertIn('data-step="0"', result)
        self.assertIn('Step 4 / 4', result)
        self.assertNotIn('https://', result)
        self.assertNotIn('<script src=', result)

    def test_rejects_multiple_rows(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            inspect_forward_pass(self.model, self.example._append(self.example))


if __name__ == "__main__":
    unittest.main()
