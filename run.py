from app import create_app
from config import Config
import os

app = create_app()

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
