# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg632::shutil.copy+yaml.safe_load
# name: shutil_yaml_primitive
# summary: Uses shutil.copy, yaml.safe_load across 2 repos
# anchor_symbols: ['shutil.copy', 'yaml.safe_load']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/automl/presets/base.py::AutoMLPreset.get_config ---
def get_config(cls, path: Optional[str] = None) -> Optional[dict]:
        """Create new config template.

        Args:
            path: Path to config.

        Returns:
            Config.

        """
        if path is None:
            path = os.path.join(base_dir, cls._default_config_path)
            with open(path) as f:
                params = yaml.safe_load(f)
            return params

        else:
            shutil.copy(os.path.join(base_dir, cls._default_config_path), path)

# --- from sberbank-ai-lab__LightAutoML::lightautoml/automl/presets/base.py::AutoMLPreset.get_config ---
def get_config(cls, path: Optional[str] = None) -> Optional[dict]:
        """Create new config template.

        Args:
            path: Path to config.

        Returns:
            Config.

        """
        if path is None:
            path = os.path.join(base_dir, cls._default_config_path)
            with open(path) as f:
                params = yaml.safe_load(f)
            return params

        else:
            shutil.copy(os.path.join(base_dir, cls._default_config_path), path)
