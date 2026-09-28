# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg610::platform.system+subprocess.run
# name: platform_subprocess_primitive
# summary: Uses platform.system, subprocess.run across 2 repos
# anchor_symbols: ['platform.system', 'subprocess.run']
# observed in 2 repos: ['InfuseAI__piperider', 'code-kern-ai__refinery']...

# --- from code-kern-ai__refinery::refinery/cli.py::update._update ---
def _update():
        if platform.system() == "Windows":
            subprocess.run(["update.bat"])
        else:
            subprocess.run(["./update"])

# --- from code-kern-ai__refinery::refinery/cli.py::stop._stop_server ---
def _stop_server():
        if platform.system() == "Windows":
            subprocess.run(["stop.bat"])
        else:
            subprocess.run(["./stop"])

# --- from InfuseAI__piperider::tests/test_compare_summary_ng.py::pbcopy_string ---
def pbcopy_string(input_string):
    try:
        import platform
        if platform.system() != "Darwin":
            return

        subprocess.run(['pbcopy'], input=input_string.encode(), check=True)
        print("String copied to clipboard.")
    except Exception:
        pass

# --- from InfuseAI__piperider::piperider_cli/utils.py::remove_link ---
def remove_link(link_path):
    try:
        if platform.system() == 'Windows':
            if os.path.exists(link_path):
                subprocess.run(["rmdir", link_path], shell=True, check=True)
        else:
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.unlink(link_path)
    except OSError as e:
        raise e
