try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with `uv sync`"
    ) from e


# ✅ FIXED IMPORT BLOCK
try:
    from ..models import AgrirlAction, AgrirlObservation
    from .agrirl_env_environment import AgriCoreEnv as AgrirlEnvironment

except (ModuleNotFoundError, ImportError):
    # fallback for direct execution (uvicorn / docker / HF)
    from models import AgrirlAction, AgrirlObservation
    from server.agrirl_env_environment import AgriCoreEnv as AgrirlEnvironment


# App creation
app = create_app(
    AgrirlEnvironment,
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
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)