# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg343::base64.b64encode+mimetypes.guess_type
# name: base64_mimetypes_primitive
# summary: Uses base64.b64encode, mimetypes.guess_type across 3 repos
# anchor_symbols: ['base64.b64encode', 'mimetypes.guess_type']
# observed in 3 repos: ['K-Dense-AI__agentic-data-scientist', 'airbnb__knowledge-repo', 'violit-dev__violit']...

# --- from airbnb__knowledge-repo::knowledge_repo/converters/html.py::HTMLConverter.base64_encode_image_mapper ---
def base64_encode_image_mapper(self, tag, url):
        if tag == 'img':
            if url in self.kp_images:
                image_data = base64.b64encode(self.kp_images[url])
                image_mimetype = mimetypes.guess_type(url)[0]
                if image_mimetype is not None:
                    return f'data:{image_mimetype};base64, ' + \
                        image_data.decode(UTF8)
        return None

# --- from violit-dev__violit::src/violit/app.py::App._resolve_favicon_href ---
def _resolve_favicon_href(self, favicon: Optional[str] = None) -> str:
        source = favicon if favicon is not None else self.app_favicon or self._default_app_icon
        if not source:
            return "data:,"

        if source.startswith(("data:", "http://", "https://", "//")):
            return source

        candidate_paths = [source]
        resolved_path = os.path.abspath(source)
        if resolved_path != source:
            candidate_paths.append(resolved_path)

        existing_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if existing_path:
            mime_type, _ = mimetypes.guess_type(existing_path)
            if not mime_type:
                suffix = Path(existing_path).suffix.lower()
                if suffix == ".ico":
                    mime_type = "image/x-icon"
                elif suffix == ".svg":
                    mime_type = "image/svg+xml"
                else:
                    mime_type = "application/octet-stream"

            with open(existing_path, "rb") as favicon_file:
                encoded = base64.b64encode(favicon_file.read()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"

        if source.startswith("/"):
            return source

        return source

# --- from K-Dense-AI__agentic-data-scientist::src/agentic_data_scientist/tools/file_ops.py::read_media_file ---
def read_media_file(path: str, working_dir: str) -> str:
    """
    Read a binary/media file (images, audio, etc.) and return base64 encoded data.

    This function follows the MCP filesystem server implementation for handling
    binary files. Returns a JSON string containing the base64 data and MIME type.

    Parameters
    ----------
    path : str
        Path to the media file (relative to working_dir or absolute)
    working_dir : str
        Working directory root for security validation

    Returns
    -------
    str
        JSON string with 'data' (base64 encoded) and 'mimeType' fields,
        or error message

    Notes
    -----
    - Only files within working_dir can be accessed
    - Returns base64 encoded binary data for transmission
    - MIME type is automatically detected from file extension
    - Supported for images, audio, video, and other binary formats
    - File size limit: 10 MB (files larger than this will be rejected)
    - Media files are NOT truncated as partial media files are broken

    Examples
    --------
    >>> result = read_media_file("image.png", "/working/dir")
    >>> import json
    >>> parsed = json.loads(result)
    >>> print(parsed["mimeType"])  # "image/png"
    >>> print(parsed["data"][:20])  # "iVBORw0KGgoAAAANSUh..."
    """
    logger.info(f"[Tool:read_media_file] Reading media file '{path}'")
    try:
        file_path = _validate_path(path, working_dir)

        if not file_path.exists():
            return f"Error: File '{path}' does not exist"

        if not file_path.is_file():
            return f"Error: '{path}' is not a file"

        # Check file size before reading (10 MB limit)
        MAX_MEDIA_SIZE = 10 * 1024 * 1024  # 10 MB in bytes
        file_size = file_path.stat().st_size
        if file_size > MAX_MEDIA_SIZE:
            size_mb = file_size / (1024 * 1024)
            return f"Error: Media file exceeds size limit of 10 MB (actual: {size_mb:.1f} MB)"

        # Read file as binary and encode to base64
        with open(file_path, "rb") as f:
            file_data = f.read()

        base64_data = base64.b64encode(file_data).decode("utf-8")

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        # Return as JSON string (compatible with ADK tool interface)
        result = {
            "data": base64_data,
            "mimeType": mime_type,
        }
        return json.dumps(result)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading media file: {e}"
