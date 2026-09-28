# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg698::subprocess.run+unittest.TestSuite+unittest.TextTestRunner
# name: subprocess_unittest_primitive
# summary: Uses subprocess.run, unittest.TestSuite, unittest.TextTestRunner across 3 repos
# anchor_symbols: ['subprocess.run', 'unittest.TestSuite', 'unittest.TextTestRunner']
# observed in 3 repos: ['JacksonWuxs__DaPy', 'annoviko__pyclustering', 'erezsh__Preql']...

# --- from erezsh__Preql::tests/__main__.py::run_test_suite ---
def run_test_suite(suit):
    tests = TESTS_SUITES[suit]
    suite = unittest.TestSuite()
    for t in tests:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(t))
    unittest.TextTestRunner().run(suite)

# --- from JacksonWuxs__DaPy::DaPy/__init__.py::_unittests ---
def _unittests():
    from unittest import TestSuite, defaultTestLoader, TextTestRunner
    _tests = TestSuite()
    for case in defaultTestLoader.discover('.', 'test_*.py'):
        _tests.addTests(case)
    tester = TextTestRunner()
    tester.run(_tests)

# --- from annoviko__pyclustering::ccore/tst/ut-runner.py::Runner.__rerun ---
def __rerun(self, output):
        failures = Runner.__get_failures(output)
        if len(failures) == 0:
            return EExitCode.failure_tests_not_found, None

        logging.info("Rerun failed tests: '%s'" % failures)

        argument = "--gtest_filter="
        for fail in failures:
            argument += ":" + fail

        result = subprocess.run([self.__executable, argument], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode('utf-8')
        print(output)

        return result.returncode, output

# --- from annoviko__pyclustering::pyclustering/tests/suite_holder.py::suite_holder.run ---
def run(self, rerun=2):
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(self.__suite)
        if result.wasSuccessful() is True:
            return result

        if len(result.errors) > 0:
            return result   # no need to restart in case of errors

        for attempt in range(rerun):
            time.sleep(1)   # sleep 1 second to change current time for random seed.

            print("\n======================================================================")
            print("Rerun failed tests (attempt: %d)." % (attempt + 1))
            print("----------------------------------------------------------------------")

            failure_suite = unittest.TestSuite()
            for failure in result.failures:
                failure_suite.addTest(failure[0])

            result = unittest.TextTestRunner(stream=sys.stdout, verbosity=3).run(failure_suite)
            if result.wasSuccessful() is True:
                return result

        return result
