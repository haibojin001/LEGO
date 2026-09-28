# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg360::flask.Flask+flask.jsonify
# name: flask_primitive
# summary: Uses flask.Flask, flask.jsonify across 2 repos
# anchor_symbols: ['flask.Flask', 'flask.jsonify']
# observed in 2 repos: ['HDI-Project__ATM', 'awslabs__gluonts']...

# --- from awslabs__gluonts::src/gluonts/shell/serve/app.py::get_base_app ---
def get_base_app(execution_params):
    app = Flask("GluonTS scoring service")

    @app.errorhandler(Exception)
    def handle_error(error) -> Tuple[str, int]:
        return traceback.format_exc(), 500

    @app.route("/ping")
    def ping() -> str:
        return ""

    @app.route("/execution-parameters")
    def execution_parameters() -> Response:
        return jsonify(execution_params)

    return app

# --- from HDI-Project__ATM::atm/api/__init__.py::create_app ---
def create_app(atm, debug=False):
    db = atm.db
    app = Flask(__name__)
    app.config['DEBUG'] = debug
    app.config['SQLALCHEMY_DATABASE_URI'] = make_absolute(db.engine.url)

    # Create the Flask-Restless API manager.
    manager = APIManager(app, flask_sqlalchemy_db=SQLAlchemy(app))

    # Allow the CORS header
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    @app.route('/api/run', methods=['POST'])
    @auto_abort((KeyError, ValueError))
    def atm_run():
        data = request.json
        run_conf = RunConfig(data)

        dataruns = atm.add_datarun(**run_conf.to_dict())
        if not isinstance(dataruns, list):
            dataruns = [dataruns]

        response = {
            'status': 200,
            'datarun_ids': [datarun.id for datarun in dataruns]
        }

        return jsonify(response)

    @app.route('/')
    def swagger():
        return redirect('/static/swagger/swagger-ui/index.html')

    # Create API endpoints, which will be available at /api/<tablename> by
    # default. Allowed HTTP methods can be specified as well.
    manager.create_api(db.Dataset, methods=['GET', 'POST'])
    manager.create_api(db.Datarun, methods=['GET'])
    manager.create_api(db.Hyperpartition, methods=['GET'])
    manager.create_api(db.Classifier, methods=['GET'])

    return app
