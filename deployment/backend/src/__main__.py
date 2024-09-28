import os
from pathlib import Path

import uvicorn

os.chdir(Path(__file__).parents[1])

config = uvicorn.Config(
    "src.main:app", host="0.0.0.0", port=8000, log_level="debug", reload=True
)
server = uvicorn.Server(config)
server.run()