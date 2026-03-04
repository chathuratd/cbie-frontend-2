import unittest
import numpy as np
from temporal_analysis import TemporalAnalyzer
from confirmation_model import ConfirmationModel

class TestMathematicalModels(unittest.TestCase):
    
    def setUp(self):
        self.temporal_analyzer = TemporalAnalyzer()
        self.confirmation_model = ConfirmationModel()

    def test_gini_coefficient_perfect_consistency(self):
        # 3 events, exactly 1 day apart -> diffs: [1.0, 1.0]
        timestamps = [
            "2023-01-01T12:00:00Z",
            "2023-01-02T12:00:00Z",
            "2023-01-03T12:00:00Z"
        ]
        gini = self.temporal_analyzer.calculate_consistency(timestamps)
        self.assertAlmostEqual(gini, 0.0, places=2)

    def test_gini_coefficient_inconsistency(self):
        # Diff intervals: 1 day, 10 days
        timestamps = [
            "2023-01-01T12:00:00Z",
            "2023-01-02T12:00:00Z",
            "2023-01-12T12:00:00Z"
        ]
        gini = self.temporal_analyzer.calculate_consistency(timestamps)
        self.assertTrue(gini > 0.0)
        self.assertTrue(gini < 1.0)
        
    def test_mann_kendall_increasing(self):
        scores = [0.1, 0.2, 0.4, 0.5, 0.8, 0.9]
        trend = self.temporal_analyzer.calculate_trend(scores)
        self.assertEqual(trend, 1.0)

    def test_mann_kendall_decreasing(self):
        scores = [0.9, 0.8, 0.5, 0.4, 0.2, 0.1]
        trend = self.temporal_analyzer.calculate_trend(scores)
        self.assertEqual(trend, -1.0)
        
    def test_mann_kendall_insufficient_data(self):
        # Less than 4 points
        scores = [0.1, 0.9, 0.2]
        trend = self.temporal_analyzer.calculate_trend(scores)
        self.assertEqual(trend, 0.0)

    def test_confirmation_score_stable(self):
        # Perfect consistency (0.0), Increasing trend (1.0), Max frequency, High credibility
        score = self.confirmation_model.calculate_core_score(
            consistency_score=0.0,   # norm = 1.0
            trend_score=1.0,         # norm = 1.0
            frequency=10,            # norm = 1.0
            max_frequency=10,
            credibility_score=0.9
        )
        # Expected weights:
        # consistency: 0.35 * 1.0 = 0.35
        # frequency:   0.25 * 1.0 = 0.25
        # trend:       0.10 * 1.0 = 0.10
        # credibility: 0.30 * 0.9 = 0.27
        # Total expected = 0.97
        self.assertAlmostEqual(score, 0.97, places=2)
        self.assertEqual(self.confirmation_model.determine_status(score), "Stable")
        
    def test_confirmation_score_noise(self):
        # Terrible consistency (1.0), Decreasing trend (-1.0), Low frequency, Low credibility
        score = self.confirmation_model.calculate_core_score(
            consistency_score=1.0,  # norm = 0.0
            trend_score=-1.0,       # norm = 0.0
            frequency=1,            # norm = 0.1
            max_frequency=10,
            credibility_score=0.2
        )
        # Expected weights:
        # consistency: 0.35 * 0.0 = 0.0
        # frequency:   0.25 * 0.1 = 0.025
        # trend:       0.10 * 0.0 = 0.0
        # credibility: 0.30 * 0.2 = 0.06
        # Total expected = 0.085
        self.assertAlmostEqual(score, 0.085, places=3)
        self.assertEqual(self.confirmation_model.determine_status(score), "Noise")

    def test_absolute_fact_bypass(self):
        # Even if the final calculated score is 0.0 (utter noise/never reinforced), 
        # absolute facts bypass threshold rules.
        self.assertEqual(self.confirmation_model.determine_status(0.01, is_fact=True), "Stable Fact")

if __name__ == '__main__':
    unittest.main()
