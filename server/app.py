try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with `uv sync`"
    ) from e

try:
    from agrirl_env.models import AgrirlAction, AgrirlObservation
    from agrirl_env.server.agrirl_env_environment import AgriCoreEnv as AgrirlEnvironment
except ModuleNotFoundError:
    from models import AgrirlAction, AgrirlObservation
    from .agrirl_env_environment import AgriCoreEnv as AgrirlEnvironment

app = create_app(
    AgrirlEnvironment,
    AgrirlAction,
    AgrirlObservation,
    env_name="agrirl_env",
    max_concurrent_envs=1,  # increase this number for more concurrent WebSocket sessions
)

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)