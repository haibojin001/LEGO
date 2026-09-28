# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg350::langchain_core.messages.AIMessage+langchain_core.messages.HumanMessage+langchain_core.messages.SystemMessage
# name: langchain_core_primitive
# summary: Uses langchain_core.messages.AIMessage, langchain_core.messages.HumanMessage, langchain_core.messages.SystemMessage across 2 repos
# anchor_symbols: ['langchain_core.messages.AIMessage', 'langchain_core.messages.HumanMessage', 'langchain_core.messages.SystemMessage']
# observed in 2 repos: ['business-science__ai-data-science-team', 'starpig1129__DATAGEN']...

# --- from starpig1129__DATAGEN::src/core/node.py::create_message ---
def create_message(message: Any, name: str) -> BaseMessage:
    """Create a BaseMessage object based on the message type."""
    if isinstance(message, dict):
        content = message.get("content", "")
        message_type = str(message.get("type", "ai")).lower()
    else:
        content = getattr(message, "content", str(message))
        message_type = str(getattr(message, "type", "ai")).lower()
        
    return HumanMessage(content=content) if message_type == "human" else AIMessage(content=content, name=name)

# --- from business-science__ai-data-science-team::ai_data_science_team/multiagents/supervisor_ds_team.py::make_supervisor_ds_team._trim_messages ---
def _trim_messages(
        msgs: Sequence[BaseMessage],
        max_messages: int = TEAM_MAX_MESSAGES,
        max_chars: int = TEAM_MAX_MESSAGE_CHARS,
    ) -> list[BaseMessage]:
        trimmed: list[BaseMessage] = []
        for m in list(msgs or [])[-max_messages:]:
            content = getattr(m, "content", "")
            if isinstance(content, str) and len(content) > max_chars:
                content = content[:max_chars] + "\n...[truncated]..."
                if isinstance(m, AIMessage):
                    m = AIMessage(
                        content=content,
                        name=getattr(m, "name", None),
                        id=getattr(m, "id", None),
                    )
                elif isinstance(m, HumanMessage):
                    m = HumanMessage(content=content, id=getattr(m, "id", None))
                elif isinstance(m, SystemMessage):
                    m = SystemMessage(content=content, id=getattr(m, "id", None))
            trimmed.append(m)
        return trimmed

# --- from starpig1129__DATAGEN::src/core/node.py::human_review_node ---
def human_review_node(state: State) -> dict[str, Any]:
    """Display current state and handle user interaction."""
    try:
        print("Current research progress:")
        print(state)
        print("\nDo you need additional analysis or modifications?")
        
        while True:
            user_input = input("Enter 'yes' to continue analysis, or 'no' to end the research: ").lower()
            if user_input in ['yes', 'no']:
                break
        
        updates: dict[str, Any] = {"last_active_agent": "human"}
        
        if user_input == 'yes':
            while True:
                req = input("Please enter your request: ").strip()
                if req:
                    updates["messages"] = [HumanMessage(content=req)]
                    updates["needs_revision"] = True
                    break
        else:
            updates["needs_revision"] = False
            updates["revision_count"] = 0
        
        return updates
        
    except Exception as e:
        logger.error(f"Error in human_review: {str(e)}", exc_info=True)
        current_messages = list(get_state_attr(state, "messages", []))
        return {"messages": current_messages + [AIMessage(content=f"Error: {str(e)}", name="human_review")]}

# --- from business-science__ai-data-science-team::ai_data_science_team/multiagents/pandas_data_analyst.py::make_pandas_data_analyst.prepare_messages ---
def prepare_messages(state: PrimaryState):
        print("---PANDAS DATA ANALYST---")
        print("*************************")
        print("---PREPARE MESSAGES---")
        msgs = state.get("messages", [])
        ui = state.get("user_instructions")
        if not msgs:
            system_hint = (
                "You are a pandas data analyst orchestrator. Route the user's question to data wrangling "
                "and optional visualization. Prefer tables unless the user clearly requests a chart."
            )
            msgs = [("system", system_hint), ("user", ui)]
        if not ui:
            for msg in reversed(msgs):
                if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
                    ui = msg.content
                    break
        normalized = []
        for msg in msgs:
            if isinstance(msg, BaseMessage):
                normalized.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role in ("user", "human"):
                    normalized.append(HumanMessage(content=content))
                elif role == "system":
                    normalized.append(SystemMessage(content=content))
                elif role in ("assistant", "ai"):
                    normalized.append(AIMessage(content=content))
                else:
                    normalized.append(HumanMessage(content=str(content)))
            else:
                normalized.append(HumanMessage(content=str(msg)))
        return {"messages": normalized, "user_instructions": ui}
