"""Manager and aggregator for fine-tuning experiment metrics and artifacts.

Loads the completed evaluation and training results from the Notebook/ directory,
providing structured data for the Evaluation Dashboard, Model Comparison UI, and API.
Preserves scientific integrity:
- Reports cross-dataset generalization limitations on SCIDOCS test set
- Accurately conveys random negatives vs BM25 hard negatives performance
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class ExperimentResultsManager:
    def __init__(self, notebook_dir: Optional[str] = None):
        self.notebook_dir = notebook_dir or settings.notebook_dir
        self.ft_dir = os.path.join(self.notebook_dir, "researchnexus-finetuning")
        self.results_dir = os.path.join(self.ft_dir, "results")
        self.model_dir = os.path.join(self.ft_dir, "minilm-scifact-finetuned")
        self.plots_dir = os.path.join(self.ft_dir, "plots")

    def get_evaluation_results(self) -> Dict[str, Any]:
        """Load evaluation_results.json."""
        eval_json_path = os.path.join(self.model_dir, "evaluation_results.json")
        if os.path.exists(eval_json_path):
            with open(eval_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Load training_history.csv with per-epoch train and val loss."""
        csv_path = os.path.join(self.model_dir, "training_history.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(self.results_dir, "training_history.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df.to_dict(orient="records")
        return []

    def get_model_comparison_table(self) -> pd.DataFrame:
        """Load and return final_comparison.csv as a DataFrame."""
        csv_path = os.path.join(self.results_dir, "final_comparison.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    def get_statistical_significance(self) -> pd.DataFrame:
        """Load paired bootstrap confidence intervals and Wilcoxon tests."""
        csv_path = os.path.join(self.results_dir, "statistical_significance.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    def get_negative_ablation(self) -> pd.DataFrame:
        """Load random negatives vs BM25 hard negatives comparison."""
        csv_path = os.path.join(self.results_dir, "negative_ablation.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    def get_weight_interpolation(self) -> pd.DataFrame:
        """Load weight interpolation (alpha 0 to 1) tradeoff analysis."""
        csv_path = os.path.join(self.results_dir, "weight_interpolation.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    def get_training_config(self) -> Dict[str, Any]:
        """Load training hyperparameters and setup metadata."""
        config_path = os.path.join(self.model_dir, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_available_plots(self) -> List[Dict[str, str]]:
        """List all generated evaluation and diagnostic plots."""
        if not os.path.exists(self.plots_dir):
            return []
        plots = []
        for f in sorted(os.listdir(self.plots_dir)):
            if f.endswith((".png", ".jpg", ".svg")):
                name = os.path.splitext(f)[0].replace("_", " ").title()
                plots.append({
                    "filename": f,
                    "title": name,
                    "path": os.path.join(self.plots_dir, f),
                })
        return plots

    def get_executive_summary(self) -> Dict[str, Any]:
        """Produce a scientifically rigorous synthesis of the experiment."""
        return {
            "model_architecture": "cross-encoder/ms-marco-MiniLM-L-6-v2 (fine-tuned with BertForSequenceClassification)",
            "dataset_in_domain": "SciFact (expert-annotated scientific claims and abstracts)",
            "dataset_out_of_domain": "SCIDOCS (scientific document recommendation and citation links)",
            "key_conclusions": [
                {
                    "topic": "In-Domain Improvement (SciFact)",
                    "result": "Statistically significant improvement on SciFact validation (+0.0417 nDCG@10, 95% CI [0.0047, 0.0784]) and SciFact test (+0.0382 nDCG@10, 95% CI [0.0135, 0.0642]).",
                    "status": "POSITIVE",
                },
                {
                    "topic": "Cross-Dataset Generalization (SCIDOCS)",
                    "result": "Performance decreased on SCIDOCS test (-0.0177 nDCG@10, 95% CI [-0.0230, -0.0124]), highlighting domain-specific adaptation trade-offs.",
                    "status": "NEGATIVE_RESULT",
                },
                {
                    "topic": "Negative Sampling Strategy",
                    "result": "BM25 hard negatives did NOT outperform random negatives (SciFact test nDCG@10: 0.7144 with hard negatives vs 0.7360 with random negatives).",
                    "status": "SURPRISING_FINDING",
                },
                {
                    "topic": "Weight Interpolation / Model Soup",
                    "result": "Interpolating model weights (alpha ~ 0.5 - 0.75) preserves in-domain gains while softening the out-of-domain degradation.",
                    "status": "MITIGATION",
                },
            ],
        }


results_manager = ExperimentResultsManager()
