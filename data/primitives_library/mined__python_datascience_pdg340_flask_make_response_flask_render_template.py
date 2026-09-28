# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg340::flask.make_response+flask.render_template
# name: flask_primitive
# summary: Uses flask.make_response, flask.render_template across 2 repos
# anchor_symbols: ['flask.make_response', 'flask.render_template']
# observed in 2 repos: ['airbnb__knowledge-repo', 'litaotao__IPython-Dashboard']...

# --- from litaotao__IPython-Dashboard::dashboard/server/resources/sql.py::Sql.get ---
def get(self):
        return make_response(render_template('sql.html', api_root=config.app_host))

# --- from litaotao__IPython-Dashboard::dashboard/server/resources/home.py::Home.get ---
def get(self):
        return make_response(render_template('home.html', api_root=config.app_host))

# --- from airbnb__knowledge-repo::knowledge_repo/app/routes/index.py::create ---
def create(knowledge_format=None):
    """ Renders the create knowledge view """
    if knowledge_format is None:
        return render_template(
            "create-knowledge.html",
            web_editor_enabled=current_app.config['WEB_EDITOR_PREFIXES'] != [])

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    knowledge_template = f'knowledge_template.{knowledge_format}'
    filename = os.path.join(cur_dir, '../../templates', knowledge_template)
    response = make_response(open(filename).read())
    response.headers["Content-Disposition"] = \
        "attachment; filename=" + knowledge_template
    return response

# --- from airbnb__knowledge-repo::knowledge_repo/app/routes/index.py::site_map_posts ---
def site_map_posts():
    posts = db_session.query(Post).filter(Post.is_published)
    values = []
    # [TODO]handle https with deployed with action gateway
    app_domain = request.url_root.replace("http", "https")
    for post in posts:
        values.append(
            {
                "loc": f"{app_domain}post/{post.path}",
                "lastmod": post.updated_at.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "priority": 1
            }
        )
    template = render_template('sitemap.xml', values=values)
    content = gzip.compress(template.encode('utf-8'))
    response = make_response(content)
    response.headers['Content-Type'] = 'application/xml'
    response.headers['Content-length'] = len(content)
    response.headers['Content-Encoding'] = 'gzip'
    return response
