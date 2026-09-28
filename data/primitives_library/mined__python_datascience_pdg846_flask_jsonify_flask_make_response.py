# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg846::flask.jsonify+flask.make_response
# name: flask_primitive
# summary: Uses flask.jsonify, flask.make_response across 2 repos
# anchor_symbols: ['flask.jsonify', 'flask.make_response']
# observed in 2 repos: ['jwkvam__bowtie', 'litaotao__IPython-Dashboard']...

# --- from jwkvam__bowtie::bowtie/_app.py::App._endpoints.gen_upload.upload ---
def upload():
                upfile = request.files['file']
                retval = func(upfile.filename, upfile.stream)
                if retval:
                    return make_response(jsonify(), 400)
                return make_response(jsonify(), 200)

# --- from jwkvam__bowtie::bowtie/_app.py::App._endpoints.gen_upload ---
def gen_upload(func) -> Callable:
            def upload():
                upfile = request.files['file']
                retval = func(upfile.filename, upfile.stream)
                if retval:
                    return make_response(jsonify(), 400)
                return make_response(jsonify(), 200)

            return upload

# --- from litaotao__IPython-Dashboard::dashboard/server/utils.py::build_response ---
def build_response(content, code=200):
    """Build response, add headers"""
    response = make_response( jsonify(content), content['code'] )
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = \
            'Origin, X-Requested-With, Content-Type, Accept, Authorization'
    return response
