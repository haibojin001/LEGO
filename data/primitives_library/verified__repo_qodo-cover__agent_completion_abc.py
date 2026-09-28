from abc import ABC, abstractmethod
from typing import Tuple


class AgentCompletionABC(ABC):
    """Abstract base class for AI-driven prompt completion implementations."""

    @abstractmethod
    def generate_tests(
        self,
        source_file_name: str,
        max_tests: int,
        source_file_numbered: str,
        code_coverage_report: str,
        language: str,
        test_file: str,
        test_file_name: str,
        testing_framework: str,
        additional_instructions_text: str = None,
        additional_includes_section: str = None,
        failed_tests_section: str = None,
    ) -> Tuple[str, int, int, str]:
        """Generate tests intended to improve source coverage."""
        pass

    @abstractmethod
    def analyze_test_failure(
        self,
        source_file_name: str,
        source_file: str,
        processed_test_file: str,
        stdout: str,
        stderr: str,
        test_file_name: str,
    ) -> Tuple[str, int, int, str]:
        """Analyze a test execution failure."""
        pass

    @abstractmethod
    def analyze_test_insert_line(
        self,
        language: str,
        test_file_numbered: str,
        test_file_name: str,
        additional_instructions_text: str = None,
    ) -> Tuple[str, int, int, str]:
        """Determine where generated tests should be inserted."""
        pass

    @abstractmethod
    def analyze_test_against_context(
        self,
        language: str,
        test_file_content: str,
        test_file_name_rel: str,
        context_files_names_rel: str,
    ) -> Tuple[str, int, int, str]:
        """Assess a test file in relation to its contextual source files."""
        pass

    @abstractmethod
    def analyze_suite_test_headers_indentation(
        self,
        language: str,
        test_file_name: str,
        test_file: str,
    ) -> Tuple[str, int, int, str]:
        """Analyze test-suite headers, indentation, and existing tests."""
        pass

    @abstractmethod
    def adapt_test_command_for_a_single_test_via_ai(
        self,
        test_file_relative_path: str,
        test_command: str,
        project_root_dir: str,
    ) -> Tuple[str, int, int, str]:
        """Adapt a test command so it runs a single test file."""
        pass