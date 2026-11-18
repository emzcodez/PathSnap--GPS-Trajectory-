"""Optimization Results Logging and Reporting for PathSnap

Author: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
Hackathon: PathSnap Distributed Map-Matching

Provides structured recording, JSON serialization, and Markdown table generation
for Bayesian optimization trials, search spaces, and tuned-vs-untuned comparisons.
"""

from typing import Dict, Any, List, Optional
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class OptimizationTrialRecord:
    trial_id: int
    params: Dict[str, Any]
    score: float
    duration_seconds: float
    state: str = "COMPLETE"


@dataclass
class OptimizationExperimentReport:
    study_name: str
    target_component: str
    objective_description: str
    search_space_description: Dict[str, Any]
    total_trials: int
    best_trial_id: int
    best_params: Dict[str, Any]
    best_score: float
    baseline_score: Optional[float]
    total_compute_seconds: float
    trials: List[Dict[str, Any]]

    def save_json(self, output_path: str) -> None:
        """Save report to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def to_markdown_summary(self) -> str:
        """Generate formatted Markdown report section for final submission."""
        gain_str = ""
        if self.baseline_score is not None:
            gain = self.best_score - self.baseline_score
            pct = (gain / max(1e-8, self.baseline_score)) * 100.0
            gain_str = f"| Baseline (Untuned) Score | {self.baseline_score:.4f} |\n" \
                       f"| Absolute Improvement | +{gain:.4f} (+{pct:.2f}%) |\n"

        md = f"""### Bayesian Hyperparameter Optimization: {self.study_name}

**Target Component:** {self.target_component}  
**Objective:** {self.objective_description}  
**Total Trials:** {self.total_trials}  
**Total Compute Cost:** {self.total_compute_seconds:.2f} seconds  

#### Tuned vs. Untuned Comparison
| Configuration | Score |
|---|---|
| Best Tuned Score | **{self.best_score:.4f}** (Trial #{self.best_trial_id}) |
{gain_str}
#### Optimal Hyperparameters
```json
{json.dumps(self.best_params, indent=2)}
```

#### Top Trials
| Trial # | Parameters | Objective Score | Duration (s) |
|---|---|---|---|
"""
        sorted_trials = sorted(self.trials, key=lambda x: x.get("score", 0.0), reverse=True)
        for t in sorted_trials[:5]:
            param_str = ", ".join(f"{k}={v}" for k, v in t.get("params", {}).items())
            md += f"| {t.get('trial_id')} | `{param_str}` | {t.get('score', 0.0):.4f} | {t.get('duration_seconds', 0.0):.2f}s |\n"

        return md
