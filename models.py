# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Agrirl Env Environment.

The agrirl_env environment simulates agricultural management.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AgrirlAction(Action):
    """Action for the Agrirl Env environment."""

    action: str = Field(..., description="Action to take: 'irrigate', 'fertilize', or 'harvest'")


class AgrirlObservation(Observation):
    """Observation from the Agrirl Env environment."""

    soil_moisture: float = Field(default=50.0, description="Current soil moisture level")
    crop_growth: float = Field(default=0.0, description="Current crop growth level")
    day: int = Field(default=1, description="Current day")
    weather: str = Field(default="sunny", description="Current weather")
    done: bool = Field(default=False, description="Whether the episode is done")
    reward: float = Field(default=0.0, description="Reward for the last action")
