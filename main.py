from flask import Flask
import os
from api.routes.collage import collage_api

def create_app():
    app = Flask(__name__)

    app_settings =  os.getenv("APP_SETTINGS")

    app.config.from_object(app_settings)

    app.register_blueprint(collage_api, url_prefix='/collage')

    return app


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app = create_app()

    app.run(host='0.0.0.0', port=port)