try:
    from . import agent
except ImportError:
    # Agent requires Google ADK which may not be available
    agent = None
