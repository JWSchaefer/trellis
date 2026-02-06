import tomllib


def load_config() -> dict:
    try:
        with open('pyproject.toml', 'rb') as f:
            config = tomllib.load(f)
            config = config.get('tool', {}).get('trellis', {})
            return config
    except Exception:
        return {}


config = load_config()
