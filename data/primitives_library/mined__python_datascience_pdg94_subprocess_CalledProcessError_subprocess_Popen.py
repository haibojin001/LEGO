# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg94::subprocess.CalledProcessError+subprocess.Popen
# name: subprocess_primitive
# summary: Uses subprocess.CalledProcessError, subprocess.Popen across 4 repos
# anchor_symbols: ['subprocess.CalledProcessError', 'subprocess.Popen']
# observed in 4 repos: ['AgnostiqHQ__covalent', 'alan-turing-institute__CleverCSV', 'jasmcaus__caer', 'piskvorky__gensim']...

# --- from AgnostiqHQ__covalent::covalent/_file_transfer/strategies/rsync_strategy.py::Rsync.return_subprocess_callable.callable ---
def callable():
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            output, error = p.communicate()
            if p.returncode != 0:
                raise CalledProcessError(p.returncode, f'"{cmd}" with error: {str(error)}')

# --- from AgnostiqHQ__covalent::covalent/_file_transfer/strategies/rsync_strategy.py::Rsync.return_subprocess_callable ---
def return_subprocess_callable(self, cmd) -> None:
        def callable():
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            output, error = p.communicate()
            if p.returncode != 0:
                raise CalledProcessError(p.returncode, f'"{cmd}" with error: {str(error)}')

        return callable

# --- from jasmcaus__caer::docs/gh-pages.py::sh2 ---
def sh2(cmd):
    """Execute command in a subshell, return stdout.
    Stderr is unbuffered from the subshell.x"""
    p = Popen(cmd, stdout=PIPE, shell=True)
    out = p.communicate()[0]
    retcode = p.returncode
    if retcode:
        print(out.rstrip())
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip()

# --- from jasmcaus__caer::docs/gh-pages.py::sh3 ---
def sh3(cmd):
    """Execute command in a subshell, return stdout, stderr
    If anything appears in stderr, print it out to sys.stderr"""
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip(), err.rstrip()

# --- from piskvorky__gensim::release/hijack_pr.py::hijack ---
def hijack(prid):
    url = f"https://api.github.com/repos/RaRe-Technologies/gensim/pulls/{prid}"
    with smart_open.open(url) as fin:
        prinfo = json.load(fin)

    user = prinfo['head']['user']['login']
    ssh_url = prinfo['head']['repo']['ssh_url']

    remotes = check_output(['git', 'remote']).split('\n')
    if user not in remotes:
        subprocess.check_call(['git', 'remote', 'add', user, ssh_url])

    subprocess.check_call(['git', 'fetch', user])

    ref = prinfo['head']['ref']
    subprocess.check_call(['git', 'checkout', f'{user}/{ref}'])

    #
    # Prefix the local branch name with the user to avoid naming clashes with
    # existing branches, e.g. develop
    #
    subprocess.check_call(['git', 'switch', '-c', f'{user}_{ref}'])

    #
    # Set the upstream so we can push back to it more easily
    #
    subprocess.check_call(['git', 'branch', '--set-upstream-to', f'{user}/{ref}'])

# --- from alan-turing-institute__CleverCSV::make_release.py::Step.execute ---
def execute(
        self, cmd: str, silent: bool = False, confirm: bool = True
    ) -> str:
        if not silent:
            color_print(f"Running: {cmd}", color="magenta", style="bright")
        if confirm:
            wait_for_enter()
        stdout = ""
        with subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as p:
            for line in p.stdout:
                stdout += line
                if not silent:
                    print(line, end="")
        if p.returncode:
            print("--- begin stdout ---")
            print(stdout)
            print("--- end stdout ---")
            raise subprocess.CalledProcessError(
                p.returncode, p.args, stdout, p.stderr
            )
        return stdout.rstrip()

# --- from piskvorky__gensim::gensim/utils.py::check_output ---
def check_output(stdout=subprocess.PIPE, *popenargs, **kwargs):
    r"""Run OS command with the given arguments and return its output as a byte string.

    Backported from Python 2.7 with a few minor modifications. Used in word2vec/glove2word2vec tests.
    Behaves very similar to https://docs.python.org/2/library/subprocess.html#subprocess.check_output.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import check_output
        >>> check_output(args=['echo', '1'])
        '1\n'

    Raises
    ------
    KeyboardInterrupt
        If Ctrl+C pressed.

    """
    try:
        logger.debug("COMMAND: %s %s", popenargs, kwargs)
        process = subprocess.Popen(stdout=stdout, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            error = subprocess.CalledProcessError(retcode, cmd)
            error.output = output
            raise error
        return output
    except KeyboardInterrupt:
        process.terminate()
        raise
