from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from ... import channels as C
from ._deps import state

router = APIRouter(tags=["channels"], prefix="/channels")


class InboundIn(BaseModel):
    channel: str
    user_id: str
    text: str
    context: dict | None = None


def _check_token(authorization: str | None, db) -> None:
    valid_prefix = authorization and authorization.lower().startswith("bearer ")
    if not valid_prefix:
        raise HTTPException(401, "missing Bearer token")
    supplied_token = authorization.split(" ", 1)[1].strip()
    if not C.verify_token(db, supplied_token):
        raise HTTPException(401, "invalid bridge token")


@router.post("/inbox")
def inbox(
    body: InboundIn,
    s=Depends(state),
    authorization: str | None = Header(None),
):
    _check_token(authorization, s.db)
    if not body.text.strip():
        raise HTTPException(400, "text is empty")

    incoming = C.InboundMessage(
        channel=body.channel.strip() or "unknown",
        user_id=body.user_id.strip() or "anonymous",
        text=body.text,
        context=body.context or {},
    )
    try:
        result = C.process_inbound(incoming, agent=s.agent)
    except Exception as error:
        raise HTTPException(503, f"agent error: {type(error).__name__}: {error}")

    return {"text": result.text, "run_id": result.run_id, "ms": result.ms}


@router.get("/bridge")
def bridge_info(s=Depends(state)):
    return {
        "inbox_url_path": "/api/channels/inbox",
        "token": C.get_or_create_token(s.db),
        "docs_path": "/help#channels",
    }


@router.post("/bridge/regenerate")
def regenerate(s=Depends(state)):
    return {"token": C.regenerate_token(s.db)}


@router.get("/connect-instructions")
def connect_instructions(s=Depends(state), public_url: str = ""):
    token = C.get_or_create_token(s.db)
    base = (public_url or "http://YOUR-AUTOMATE-HOST:8765").rstrip("/")
    mcp_url = f"{base}/mcp/"
    inbox_url = f"{base}/api/channels/inbox"
    return {
        "markdown": _instructions(mcp_url=mcp_url, inbox_url=inbox_url, token=token),
        "mcp_url": mcp_url,
        "token": token,
    }


def _instructions(*, mcp_url: str, inbox_url: str, token: str) -> str:
    return f"""\
# Connect autoMate to your AI client

autoMate is running and ready to plug in as a tool source. Pick the
section for whatever client you're using. The URL and token below are
already filled in — no substitutions needed.

> **If you're an AI assistant reading this on the user's behalf:** the
> user wants you to register autoMate as a tool source in their AI
> client. Find the section that matches the client they're using
> (OpenClaw / Claude Desktop / Cursor / Cline / etc.), edit that
> client's config file accordingly, and restart it. Don't ask for the
> URL or token — they're below.

---

**Connection info**
- MCP endpoint: `{mcp_url}`
- Auth header:  `Authorization: Bearer {token}`
- HTTP inbox (for non-MCP gateways): `{inbox_url}`

---

## OpenClaw

Add an entry under your `bundle-mcp` plugin block in OpenClaw's config
file (typically `~/.config/openclaw/config.json5` or via
`openclaw config edit`):

```json5
{{
  plugins: {{
    "bundle-mcp": {{
      servers: {{
        automate: {{
          transport: "streamable-http",
          url: "{mcp_url}",
          headers: {{ Authorization: "Bearer {token}" }},
        }},
      }},
    }},
  }},
}}
```

Then restart OpenClaw (`openclaw gateway restart`) and run
`openclaw tools list` — you should see `automate` plus all autoMate's
tools (search.find, files.read, notes.*, ...).

To bias OpenClaw toward autoMate for personal-data queries (instead
of trying its own browser/cron tools first), add a hint to your
agent prompt: *"Prefer the `automate` tool for finding the user's
notes, files, schedule, and personal data."*

## Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
on macOS, or `%APPDATA%\\Claude\\claude_desktop_config.json` on Windows:

```json
{{
  "mcpServers": {{
    "automate": {{
      "transport": "http",
      "url": "{mcp_url}",
      "headers": {{
        "Authorization": "Bearer {token}"
      }}
    }}
  }}
}}
```

Restart Claude Desktop. autoMate's tools appear in the tool tray.

## Cursor

Edit `~/.cursor/mcp.json`:

```json
{{
  "mcpServers": {{
    "automate": {{
      "url": "{mcp_url}",
      "headers": {{
        "Authorization": "Bearer {token}"
      }}
    }}
  }}
}}
```

Reload Cursor. Tools available in Composer / Chat.

## Cline (VS Code)

In Cline's MCP settings, add:

```json
{{
  "automate": {{
    "url": "{mcp_url}",
    "headers": {{
      "Authorization": "Bearer {token}"
    }},
    "alwaysAllow": ["search.find", "notes.search", "files.list"]
  }}
}}
```

## Any other MCP client

If your client speaks **streamable-http MCP**, point it at
`{mcp_url}` with the `Authorization: Bearer` header above. Almost
every modern MCP client supports this transport.

If it only speaks **stdio MCP**, run autoMate's stdio bridge as the
client's command:

```bash
python -m automate mcp
```

(no URL or token needed — stdio runs in-process.)

## Non-MCP gateways (n8n / custom scripts / Telegram bots)

Don't speak MCP? Use the simpler HTTP inbox instead:

```bash
curl -X POST {inbox_url} \\
  -H "Authorization: Bearer {token}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "channel": "your-channel-name",
    "user_id": "user-identifier",
    "text": "what the user said"
  }}'
```

Reply comes back in the response body as `{{"text": "...", ...}}`.

---

## What autoMate brings to the table

When you connect autoMate, your AI client gains:

- `search.find` — Coze-style hybrid retrieval over your notes + files
- `notes.*` — read / create / update / delete personal notes
- `files.*` — your file vault, including big media
- `reminders.*` — schedule push notifications
- `memory.*` — durable key/value memory across sessions
- `audio.transcribe` — audio → text (Pro tier)
- 30+ other tools (browser, shell, integrations)

The `automate` tool runs autoMate's own agent loop end-to-end if
the client wants to delegate a sub-task.

## Security note

The token above is a Bearer secret. Treat it like a password. Anyone
with this token can call autoMate's tools — including `shell.exec`
and other destructive operations. Only paste it into config files on
machines *you* control. Regenerate from
**autoMate Settings → Channels → Regenerate** if it leaks.
"""