import math
from typing import Dict, List
from pydantic import BaseModel

class ModelElo(BaseModel):
    model_name: str
    elo: float = 1000.0
    wins: int = 0
    losses: int = 0
    ties: int = 0

class EloRatingEngine:
    def __init__(self, k_factor: float = 32.0):
        self.k_factor = k_factor
        self.ratings: Dict[str, ModelElo] = {}

    def get_or_create(self, name: str) -> ModelElo:
        if name not in self.ratings:
            self.ratings[name] = ModelElo(model_name=name)
        return self.ratings[name]

    def record_match(self, a: str, b: str, winner: str):
        pa = self.get_or_create(a)
        pb = self.get_or_create(b)
        exp_a = 1.0 / (1.0 + math.pow(10, (pb.elo - pa.elo) / 400.0))
        act_a = 1.0 if winner == "model_a" else (0.5 if winner == "tie" else 0.0)
        pa.elo += self.k_factor * (act_a - exp_a)
        pb.elo += self.k_factor * ((1.0 - act_a) - (1.0 - exp_a))

# K-factor decay
