from pydantic import BaseModel
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openenv.core.env_server.types import Action, Observation
else:
    Action = BaseModel
    Observation = BaseModel


class AgrirlAction(Action):
    action: Literal["wait", "irrigate", "fertilize", "harvest"]


class AgrirlObservation(Observation):
    soil_moisture: float
    crop_growth: float
    day: int
    weather: Literal["sunny", "rainy", "cloudy"]
    done: bool = False
    reward: float = 0.0
    score: Optional[float] = None   
    task: Optional[str] = None