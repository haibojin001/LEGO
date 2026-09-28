# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg253::subprocess.Popen+subprocess.STARTUPINFO+sys.exc_info
# name: subprocess_sys_primitive
# summary: Uses subprocess.Popen, subprocess.STARTUPINFO, sys.exc_info across 2 repos
# anchor_symbols: ['subprocess.Popen', 'subprocess.STARTUPINFO', 'sys.exc_info']
# observed in 2 repos: ['modin-project__modin', 'xorbitsai__xorbits']...

# --- from xorbitsai__xorbits::python/versioneer.py::run_command ---
def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process = None

    popen_kwargs = {}
    if sys.platform == "win32":
        # This hides the console window if pythonw.exe is used
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs["startupinfo"] = startupinfo

    for command in commands:
        try:
            dispcmd = str([command] + args)
            # remember shell=False, so use git.cmd on windows, not just git
            process = subprocess.Popen(
                [command] + args,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=(subprocess.PIPE if hide_stderr else None),
                **popen_kwargs,
            )
            break
        except OSError:
            e = sys.exc_info()[1]
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print("unable to run %s" % dispcmd)
                print(e)
            return None, None
    else:
        if verbose:
            print("unable to find command, tried %s" % (commands,))
        return None, None
    stdout = process.communicate()[0].strip().decode()
    if process.returncode != 0:
        if verbose:
            print("unable to run %s (error)" % dispcmd)
            print("stdout was %s" % stdout)
        return None, process.returncode
    return stdout, process.returncode

# --- from xorbitsai__xorbits::python/xorbits/_version.py::run_command ---
def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process = None

    popen_kwargs = {}
    if sys.platform == "win32":
        # This hides the console window if pythonw.exe is used
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs["startupinfo"] = startupinfo

    for command in commands:
        try:
            dispcmd = str([command] + args)
            # remember shell=False, so use git.cmd on windows, not just git
            process = subprocess.Popen(
                [command] + args,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=(subprocess.PIPE if hide_stderr else None),
                **popen_kwargs,
            )
            break
        except OSError:
            e = sys.exc_info()[1]
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print("unable to run %s" % dispcmd)
                print(e)
            return None, None
    else:
        if verbose:
            print("unable to find command, tried %s" % (commands,))
        return None, None
    stdout = process.communicate()[0].strip().decode()
    if process.returncode != 0:
        if verbose:
            print("unable to run %s (error)" % dispcmd)
            print("stdout was %s" % stdout)
        return None, process.returncode
    return stdout, process.returncode

# --- from modin-project__modin::versioneer.py::run_command ---
def run_command(
    commands: List[str],
    args: List[str],
    cwd: Optional[str] = None,
    verbose: bool = False,
    hide_stderr: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process = None

    popen_kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        # This hides the console window if pythonw.exe is used
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs["startupinfo"] = startupinfo

    for command in commands:
        try:
            dispcmd = str([command] + args)
            # remember shell=False, so use git.cmd on windows, not just git
            process = subprocess.Popen(
                [command] + args,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=(subprocess.PIPE if hide_stderr else None),
                **popen_kwargs,
            )
            break
        except OSError as e:
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print("unable to run %s" % dispcmd)
                print(e)
            return None, None
    else:
        if verbose:
            print("unable to find command, tried %s" % (commands,))
        return None, None
    stdout = process.communicate()[0].strip().decode()
    if process.returncode != 0:
        if verbose:
            print("unable to run %s (error)" % dispcmd)
            print("stdout was %s" % stdout)
        return None, process.returncode
    return stdout, process.returncode

# --- from modin-project__modin::modin/_version.py::run_command ---
def run_command(
    commands: List[str],
    args: List[str],
    cwd: Optional[str] = None,
    verbose: bool = False,
    hide_stderr: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """Call the given command(s)."""
    assert isinstance(commands, list)
    process = None

    popen_kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        # This hides the console window if pythonw.exe is used
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        popen_kwargs["startupinfo"] = startupinfo

    for command in commands:
        try:
            dispcmd = str([command] + args)
            # remember shell=False, so use git.cmd on windows, not just git
            process = subprocess.Popen(
                [command] + args,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=(subprocess.PIPE if hide_stderr else None),
                **popen_kwargs,
            )
            break
        except OSError as e:
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print("unable to run %s" % dispcmd)
                print(e)
            return None, None
    else:
        if verbose:
            print("unable to find command, tried %s" % (commands,))
        return None, None
    stdout = process.communicate()[0].strip().decode()
    if process.returncode != 0:
        if verbose:
            print("unable to run %s (error)" % dispcmd)
            print("stdout was %s" % stdout)
        return None, process.returncode
    return stdout, process.returncode
