from pydantic import BaseModel, Field
from typing import List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openenv.core.env_server.types import Action, Observation
else:
    Action = BaseModel
    Observation = BaseModel

WeatherType = Literal["sunny", "rainy", "cloudy", "heatwave", "frost"]

class Crop(BaseModel):
    id: int
    moisture: float
    growth: float
    stage: Literal["seed", "vegetative", "flowering", "mature"]
    wait_days: int
    fertilized_times: int
    pest_level: float

class AgrirlAction(Action):
    crop_id: int=0
    action: Literal["irrigate", "fertilize", "wait", "harvest", "pesticide"]

class AgrirlObservation(Observation):
    crops: List[Crop]
    water: float
    fertilizer: float
    pesticide: float
    energy: float
    day: int
    weather: WeatherType
    forecast: WeatherType
    market_price: float
    soil_health: float
    reward: float = 0.0
    done: bool = False
    score: Optional[float] = None