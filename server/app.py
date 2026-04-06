try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with `uv sync`"
    ) from e


try:
    from ..models import AgrirlAction, AgrirlObservation
    from .agrirl_env_environment import AgriCoreEnv 
except (ModuleNotFoundError, ImportError):
    from models import AgrirlAction, AgrirlObservation
    from server.agrirl_env_environment import AgriCoreEnv 

app = create_app(
    AgriCoreEnv,
    AgrirlAction,
    AgrirlObservation,
    env_name="agrirl_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)