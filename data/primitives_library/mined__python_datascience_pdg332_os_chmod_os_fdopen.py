# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg332::os.chmod+os.fdopen
# name: os_primitive
# summary: Uses os.chmod, os.fdopen across 2 repos
# anchor_symbols: ['os.chmod', 'os.fdopen']
# observed in 2 repos: ['mahmoud__boltons', 'run-house__kubetorch']...

# --- from run-house__kubetorch::python_client/tests/utils.py::write_temp_file_fn ---
def write_temp_file_fn(temp_fd, temp_path, fn_contents):
    # temp_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with os.fdopen(temp_fd, "w") as temp_file:
        temp_file.write(fn_contents)
        temp_file.flush()
    os.chmod(temp_path, 0o644)

# --- from mahmoud__boltons::boltons/fileutils.py::AtomicSaver._open_part_file ---
def _open_part_file(self):
        do_chmod = True
        file_perms = self.file_perms
        if file_perms is None:
            try:
                # try to copy from file being replaced
                stat_res = os.stat(self.dest_path)
                file_perms = stat.S_IMODE(stat_res.st_mode)
            except OSError:
                # default if no destination file exists
                file_perms = self._default_file_perms
                do_chmod = False  # respect the umask

        fd = os.open(self.part_path, self.open_flags, file_perms)
        set_cloexec(fd)
        self.part_file = os.fdopen(fd, self.mode, self.buffering)

        # if default perms are overridden by the user or previous dest_path
        # chmod away the effects of the umask
        if do_chmod:
            try:
                os.chmod(self.part_path, file_perms)
            except OSError:
                self.part_file.close()
                raise
        return
