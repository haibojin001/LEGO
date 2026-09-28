# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg798::tempfile.NamedTemporaryFile+textwrap.dedent
# name: tempfile_textwrap_primitive
# summary: Uses tempfile.NamedTemporaryFile, textwrap.dedent across 2 repos
# anchor_symbols: ['tempfile.NamedTemporaryFile', 'textwrap.dedent']
# observed in 2 repos: ['rpy2__rpy2', 'turicas__rows']...

# --- from turicas__rows::tests/tests_utils.py::SchemaTestCase.test_load_schema ---
def test_load_schema(self):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.files_to_delete.append(temp.name)
        temp.file.write(
            dedent(
                """
        field_name,field_type
        f1,text
        f2,decimal
        f3,float
        f4,integer
        """
            )
            .strip()
            .encode("utf-8")
        )
        temp.file.close()
        schema = rows.utils.load_schema(temp.name)
        expected = OrderedDict(
            [
                ("f1", fields.TextField),
                ("f2", fields.DecimalField),
                ("f3", fields.FloatField),
                ("f4", fields.IntegerField),
            ]
        )
        assert schema == expected

# --- from turicas__rows::tests/tests_plugin_postgresql.py::PluginPostgreSQLTestCase.test_pgimport_force_null ---
def test_pgimport_force_null(self):
        temp = tempfile.NamedTemporaryFile()
        filename = "{}.csv".format(temp.name)
        temp.close()
        self.files_to_delete.append(filename)
        with open(filename, mode="wb") as fobj:
            fobj.write(
                dedent(
                    """
                field1,field2
                "","4"
                ,2
                """
                )
                .strip()
                .encode("utf-8")
            )
        rows.utils.pgimport(
            filename=filename,
            database_uri=TEST_DATABASE_URL,
            table_name="rows_force_null",
        )
        table = rows.import_from_postgresql(TEST_DATABASE_URL, "rows_force_null")
        assert table[0].field1 is None
        assert table[0].field2 == 4
        assert table[1].field1 is None
        assert table[1].field2 == 2

# --- from rpy2__rpy2::rpy2-rinterface/src/rpy2/rinterface/tests/test_embedded_r.py::test_interrupt_r ---
def test_interrupt_r(rcode):
    expected_code = 42  # this is an arbitrary exit code that we check for below
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                     delete=False) as rpy_code:
        rpy2_path = os.path.dirname(rpy2.__path__[0])

        rpy_code_str = textwrap.dedent("""
        import sys
        sys.path.insert(0, '%s')
        import rpy2.rinterface as ri
        from rpy2.rinterface_lib import callbacks
        from rpy2.rinterface_lib import embedded

        ri.initr()
        def f(x):
            # This flush is important to make sure we avoid a deadlock.
            print(x, flush=True)
        rcode = '''
        message('executing-rcode')
        console.flush()
        %s
        '''
        with callbacks.obj_in_module(callbacks, 'consolewrite_print', f):
            try:
                ri.baseenv['eval'](ri.parse(rcode))
            except embedded.RRuntimeError:
                sys.exit(%d)
      """) % (rpy2_path, rcode, expected_code)

        rpy_code.write(rpy_code_str)
    cmd = (sys.executable, rpy_code.name)
    with open(os.devnull, 'w') as fnull:
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        # This context manager ensures that appropriate cleanup happens for the
        # process and for stdout.
        with subprocess.Popen(cmd,
                              # Since we are reading from stdout, it's important to ensure we work
                              # well with buffering. We make sure to flush when printing, but a viable
                              # alternative is starting the Python process as unbuffered using `-u`.
                              # If we weren't explicitly flushing then buffering could result in a
                              # deadlock where the parent waits for the message from the child, but the
                              # child is in an infinite loop that will only terminate if interrupted by
                              # the parent.
                              stdout=subprocess.PIPE,
                              stderr=fnull,
                              creationflags=creationflags) as child_proc:
            # We wait for the child process to send a message signalling that
            # R code is being executed since we want to ensure that we only send
            # the signal at that point. If we send it while Python is executing,
            # we would instead get a KeyboardInterrupt.
            for line in child_proc.stdout:
                if line == b'executing-rcode\n':
                    break
            sigint = signal.CTRL_C_EVENT if os.name == 'nt' else signal.SIGINT
            child_proc.send_signal(sigint)
            # Wait for the process to terminate. Timeout ensures we don't wait indefinitely.
            ret_code = child_proc.wait(timeout=10)
    # This test checks for a specific exit code to ensure that the above code
    # block exited correctly. This is important to distinguish our expected
    # process interruption from other errors the test might encounter.
    assert ret_code == expected_code
