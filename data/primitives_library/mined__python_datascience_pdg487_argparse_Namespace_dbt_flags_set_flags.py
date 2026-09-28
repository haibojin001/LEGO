# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg487::argparse.Namespace+dbt.flags.set_flags
# name: argparse_dbt_primitive
# summary: Uses argparse.Namespace, dbt.flags.set_flags across 2 repos
# anchor_symbols: ['argparse.Namespace', 'dbt.flags.set_flags']
# observed in 2 repos: ['InfuseAI__piperider', 'datafold__data-diff']...

# --- from datafold__data-diff::data_diff/dbt_parser.py::try_set_dbt_flags ---
def try_set_dbt_flags() -> None:
    try:
        from dbt.flags import set_flags

        set_flags(Namespace(MACRO_DEBUGGING=False))
    except:
        pass

# --- from InfuseAI__piperider::piperider_cli/dbt/list_task.py::_DbtListTask.__init__ ---
def __init__(self):
        self.config = _RuntimeConfig()
        self.args = make_flag()
        self.previous_state = None

        if dbt_version >= '1.5' and hasattr(flags_module, 'set_flags'):
            flags_module.set_flags(self.args)

        # The graph compiler tries to make directories when it initialized itself
        # setattr(self.args, "target_path", "/tmp/piperider-list-task/target_path")
        setattr(self.args, "target_path", 'target')
        setattr(
            self.args,
            "packages_install_path",
            "/tmp/piperider-list-task/packages_install_path",
        )

        # Args for ListTask
        setattr(self.args, "WRITE_JSON", None)
        setattr(self.args, "exclude", None)
        setattr(self.args, "output", "selector")
        setattr(self.args, "models", None)
        setattr(self.args, "INDIRECT_SELECTION", "eager")
        setattr(self.args, "WARN_ERROR", False)
        _configure_warn_error_options(self.args)
        self.args.args = argparse.Namespace()
        self.args.args.cls = ListTask

        # All nodes the compiler building
        self.node_results = []

        try:
            from dbt.task.contextvars import cv_project_root
            if self.config:
                cv_project_root.set(self.config.project_root)
        except Exception:
            # cv_project_root start to be defined since dbt-core v1.5.2
            pass
