# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agrirl Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AgrirlAction, AgrirlObservation


class AgrirlEnv(
    EnvClient[AgrirlAction, AgrirlObservation, State]
):
    """
    Client for the Agrirl Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with AgrirlEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(AgrirlAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AgrirlEnv.from_docker_image("agrirl_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(AgrirlAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: AgrirlAction) -> Dict:
        """
        Convert AgrirlAction to JSON payload for step message.

        Args:
            action: AgrirlAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AgrirlObservation]:
        """
        Parse server response into StepResult[AgrirlObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with AgrirlObservation
        """
        obs_data = payload.get("observation", {})
        observation = AgrirlObservation(
            soil_moisture=obs_data.get("soil_moisture", 50.0),
            crop_growth=obs_data.get("crop_growth", 0.0),
            day=obs_data.get("day", 1),
            weather=obs_data.get("weather", "sunny"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
