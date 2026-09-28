# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg313::os.system+platform.system
# name: os_platform_primitive
# summary: Uses os.system, platform.system across 2 repos
# anchor_symbols: ['os.system', 'platform.system']
# observed in 2 repos: ['ndleah__python-mini-project', 'vaexio__vaex']...

# --- from ndleah__python-mini-project::Morse_code_beep/main.py::play_sound ---
def play_sound(duration):
    # For Windows
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(1000, duration)  # Beep at 1000 Hz for 'duration' milliseconds
    # For Linux/macOS
    else:
        import os
        os.system('printf "\a"')  # Produces system beep

# --- from vaexio__vaex::packages/vaex-core/vaex/utils.py::os_open ---
def os_open(document):
    """Open document by the default handler of the OS, could be a url opened by a browser, a text file by an editor etc"""
    osname = platform.system().lower()
    if osname == "darwin":
        os.system("open \"" + document + "\"")
    if osname == "linux":
        cmd = "xdg-open \"" + document + "\"&"
        os.system(cmd)
    if osname == "windows":
        os.system("start \"" + document + "\"")
