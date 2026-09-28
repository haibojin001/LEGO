import logging

from backend.app_factory import create_app

app = create_app(mode="web")

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import sys

    import uvicorn

    port = 8000

    if len(sys.argv) > 1:
        for index, value in enumerate(sys.argv):
            if value == "--port" and index + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[index + 1])
                except ValueError:
                    logger.error("无效的端口号: %s", sys.argv[index + 1])
                    port = 8000

    logger.info("启动服务器，端口: %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)