from fastapi import applications
from fastapi.openapi.docs import get_redoc_html
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.docs import get_swagger_ui_oauth2_redirect_html


def register_offline_docs(app: applications):
    def custom_swagger_ui_html(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/swagger-ui/favicon.png"
        )

    def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    def redoc_html(*args, **kwargs):
        return get_redoc_html(
            *args,
            **kwargs,
            redoc_js_url="/static/redoc/bundles/redoc.standalone.js",
            redoc_favicon_url="/static/redoc/img/favicon.png",
        )

    app.get_swagger_ui_html = custom_swagger_ui_html
    app.get_swagger_ui_oauth2_redirect_html = swagger_ui_redirect
    app.get_redoc_html = redoc_html
