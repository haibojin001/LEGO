import json
from typing import Any

from .explorer import Explorer
from .template import read_template, render_template

PLAYGROUND_HTML = read_template("playground.html")

SettingsDict = dict[str, str | int | bool | dict[str, str]]


class ExplorerPlayground(Explorer):
    def __init__(
        self,
        title: str = "Ariadne GraphQL",
        share_enabled: bool = False,
        editor_cursor_shape: str | None = None,
        editor_font_family: str | None = None,
        editor_font_size: int | None = None,
        editor_reuse_headers: bool | None = None,
        editor_theme: str | None = None,
        general_beta_updates: bool | None = None,
        prettier_print_width: int | None = None,
        prettier_tab_width: int | None = None,
        prettier_use_tabs: bool | None = None,
        request_credentials: str | None = None,
        request_global_headers: dict[str, str] | None = None,
        schema_polling_enable: bool | None = None,
        schema_polling_endpoint_filter: str | None = None,
        schema_polling_interval: int | None = None,
        schema_disable_comments: bool | None = None,
        tracing_hide_tracing_response: bool | None = None,
        tracing_tracing_supported: bool | None = None,
        query_plan_hide_query_plan_response: bool | None = None,
    ) -> None:
        settings = self.build_settings(
            editor_cursor_shape=editor_cursor_shape,
            editor_font_family=editor_font_family,
            editor_font_size=editor_font_size,
            editor_reuse_headers=editor_reuse_headers,
            editor_theme=editor_theme,
            general_beta_updates=general_beta_updates,
            prettier_print_width=prettier_print_width,
            prettier_tab_width=prettier_tab_width,
            prettier_use_tabs=prettier_use_tabs,
            request_credentials=request_credentials,
            request_global_headers=request_global_headers,
            schema_polling_enable=schema_polling_enable,
            schema_polling_endpoint_filter=schema_polling_endpoint_filter,
            schema_polling_interval=schema_polling_interval,
            schema_disable_comments=schema_disable_comments,
            tracing_hide_tracing_response=tracing_hide_tracing_response,
            tracing_tracing_supported=tracing_tracing_supported,
            query_plan_hide_query_plan_response=query_plan_hide_query_plan_response,
        )

        self.parsed_html = render_template(
            PLAYGROUND_HTML,
            {
                "title": title,
                "settings": json.dumps(settings) if settings else None,
                "share_enabled": str(share_enabled).lower(),
            },
        )

    def build_settings(
        self,
        editor_cursor_shape: str | None = None,
        editor_font_family: str | None = None,
        editor_font_size: int | None = None,
        editor_reuse_headers: bool | None = None,
        editor_theme: str | None = None,
        general_beta_updates: bool | None = None,
        prettier_print_width: int | None = None,
        prettier_tab_width: int | None = None,
        prettier_use_tabs: bool | None = None,
        request_credentials: str | None = None,
        request_global_headers: dict[str, str] | None = None,
        schema_polling_enable: bool | None = None,
        schema_polling_endpoint_filter: str | None = None,
        schema_polling_interval: int | None = None,
        schema_disable_comments: bool | None = None,
        tracing_hide_tracing_response: bool | None = None,
        tracing_tracing_supported: bool | None = None,
        query_plan_hide_query_plan_response: bool | None = None,
    ) -> SettingsDict:
        result: SettingsDict = {}

        if editor_cursor_shape:
            result["editor.cursorShape"] = editor_cursor_shape
        if editor_font_family:
            result["editor.fontFamily"] = editor_font_family
        if editor_font_size:
            result["editor.fontSize"] = editor_font_size
        if editor_reuse_headers is not None:
            result["editor.reuseHeaders"] = editor_reuse_headers
        if editor_theme:
            result["editor.theme"] = editor_theme
        if general_beta_updates is not None:
            result["general.betaUpdates"] = general_beta_updates
        if prettier_print_width:
            result["prettier.printWidth"] = prettier_print_width
        if prettier_tab_width:
            result["prettier.tabWidth"] = prettier_tab_width
        if prettier_use_tabs is not None:
            result["prettier.useTabs"] = prettier_use_tabs
        if request_credentials:
            result["request.credentials"] = request_credentials
        if request_global_headers:
            result["request.globalHeaders"] = request_global_headers
        if schema_polling_enable is not None:
            result["schema.polling.enable"] = schema_polling_enable
        if schema_polling_endpoint_filter:
            result["schema.polling.endpointFilter"] = schema_polling_endpoint_filter
        if schema_polling_interval:
            result["schema.polling.interval"] = schema_polling_interval
        if schema_disable_comments is not None:
            result["schema.disableComments"] = schema_disable_comments
        if tracing_hide_tracing_response is not None:
            result["tracing.hideTracingResponse"] = tracing_hide_tracing_response
        if tracing_tracing_supported is not None:
            result["tracing.tracingSupported"] = tracing_tracing_supported
        if query_plan_hide_query_plan_response is not None:
            result["queryPlan.hideQueryPlanResponse"] = (
                query_plan_hide_query_plan_response
            )

        return result

    def html(self, request: Any) -> str:
        return self.parsed_html