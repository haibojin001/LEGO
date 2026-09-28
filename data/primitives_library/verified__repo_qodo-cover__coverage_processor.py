import csv
import json
import os
import re
import xml.etree.ElementTree as ET

from typing import List, Optional, Tuple, Union

from cover_agent.custom_logger import CustomLogger
from cover_agent.settings.config_schema import CoverageType


class CoverageProcessor:
    def __init__(
        self,
        file_path: str,
        src_file_path: str,
        coverage_type: CoverageType,
        use_report_coverage_feature_flag: bool = False,
        diff_coverage_report_path: str = None,
        logger: Optional[CustomLogger] = None,
        generate_log_files: bool = True,
    ):
        self.file_path = file_path
        self.src_file_path = src_file_path
        self.coverage_type = coverage_type
        self.logger = logger or CustomLogger.get_logger(
            __name__, generate_log_files=generate_log_files
        )
        self.use_report_coverage_feature_flag = use_report_coverage_feature_flag
        self.diff_coverage_report_path = diff_coverage_report_path

    def process_coverage_report(self, time_of_test_command: int) -> Tuple[list, list, float]:
        self.verify_report_update(time_of_test_command)
        return self.parse_coverage_report()

    def verify_report_update(self, time_of_test_command: int):
        assert os.path.exists(self.file_path), (
            f'Fatal: Coverage report "{self.file_path}" was not generated.'
        )

        modified_at = int(round(os.path.getmtime(self.file_path) * 1000))
        if modified_at <= time_of_test_command:
            self.logger.warning(
                "The coverage report file was not updated after the test command. "
                f"file_mod_time_ms: {modified_at}, "
                f"time_of_test_command: {time_of_test_command}. "
                f"{modified_at > time_of_test_command}"
            )

    def parse_coverage_report(self) -> Tuple[list, list, float]:
        if self.use_report_coverage_feature_flag:
            if self.coverage_type == "cobertura":
                return self.parse_coverage_report_cobertura()
            if self.coverage_type == "lcov":
                return self.parse_coverage_report_lcov()
            if self.coverage_type == "jacoco":
                return self.parse_coverage_report_jacoco()
            raise ValueError(f"Unsupported coverage report type: {self.coverage_type}")

        if self.coverage_type == "cobertura":
            return self.parse_coverage_report_cobertura(
                filename=os.path.basename(self.src_file_path)
            )
        if self.coverage_type == "lcov":
            return self.parse_coverage_report_lcov()
        if self.coverage_type == "jacoco":
            return self.parse_coverage_report_jacoco()
        if self.coverage_type == "diff_cover_json":
            return self.parse_json_diff_coverage_report()

        raise ValueError(f"Unsupported coverage report type: {self.coverage_type}")

    @staticmethod
    def _coverage_result(covered: list, missed: list) -> Tuple[list, list, float]:
        covered_set = set(covered)
        missed_set = set(missed) - covered_set
        total = len(covered_set) + len(missed_set)
        percentage = len(covered_set) / total if total else 0
        return list(covered_set), list(missed_set), percentage

    def parse_coverage_report_cobertura(
        self, filename: str = None
    ) -> Union[Tuple[list, list, float], dict]:
        root = ET.parse(self.file_path).getroot()

        if filename:
            covered = []
            missed = []
            for class_element in root.findall(".//class"):
                reported_name = class_element.get("filename")
                if reported_name and reported_name.endswith(filename):
                    class_covered, class_missed, _ = self.parse_coverage_data_for_class(
                        class_element
                    )
                    covered.extend(class_covered)
                    missed.extend(class_missed)
            return self._coverage_result(covered, missed)

        grouped_lines = {}
        for class_element in root.findall(".//class"):
            reported_name = class_element.get("filename")
            if not reported_name:
                continue

            class_covered, class_missed, _ = self.parse_coverage_data_for_class(
                class_element
            )
            if reported_name not in grouped_lines:
                grouped_lines[reported_name] = ([], [])
            grouped_lines[reported_name][0].extend(class_covered)
            grouped_lines[reported_name][1].extend(class_missed)

        return {
            reported_name: self._coverage_result(covered, missed)
            for reported_name, (covered, missed) in grouped_lines.items()
        }

    def parse_coverage_data_for_class(self, cls) -> Tuple[list, list, float]:
        covered = []
        missed = []

        for line in cls.findall(".//line"):
            number = int(line.get("number"))
            hits = int(line.get("hits"))
            if hits > 0:
                covered.append(number)
            else:
                missed.append(number)

        total = len(covered) + len(missed)
        return covered, missed, len(covered) / total if total else 0

    def _is_requested_file(self, report_file_name: str) -> bool:
        if not report_file_name:
            return False

        source_name = self.src_file_path
        source_basename = os.path.basename(source_name)
        normalized_report_name = report_file_name.replace("\\", "/")
        normalized_source_name = source_name.replace("\\", "/")

        return (
            normalized_report_name == normalized_source_name
            or normalized_source_name.endswith(normalized_report_name)
            or normalized_report_name.endswith(normalized_source_name)
            or os.path.basename(normalized_report_name) == source_basename
        )

    def parse_coverage_report_lcov(self):
        file_lines = {}
        active_file = None

        with open(self.file_path, "r") as coverage_file:
            for raw_line in coverage_file:
                line = raw_line.strip()

                if line.startswith("SF:"):
                    active_file = line[3:].strip()
                    if active_file and active_file not in file_lines:
                        file_lines[active_file] = ([], [])
                    continue

                if not active_file or not line.startswith("DA:"):
                    continue

                data = line[3:].split(",")
                if len(data) < 2:
                    continue

                try:
                    line_number = int(data[0])
                    hits = int(data[1])
                except ValueError:
                    continue

                covered, missed = file_lines[active_file]
                if hits > 0:
                    covered.append(line_number)
                else:
                    missed.append(line_number)

        if self.use_report_coverage_feature_flag:
            return {
                report_name: self._coverage_result(covered, missed)
                for report_name, (covered, missed) in file_lines.items()
            }

        covered = []
        missed = []
        for report_name, (file_covered, file_missed) in file_lines.items():
            if self._is_requested_file(report_name):
                covered.extend(file_covered)
                missed.extend(file_missed)

        return self._coverage_result(covered, missed)

    def parse_coverage_report_jacoco(self):
        root = ET.parse(self.file_path).getroot()
        file_lines = {}

        for source_file in root.findall(".//sourcefile"):
            source_name = source_file.get("name")
            if not source_name:
                continue

            if source_name not in file_lines:
                file_lines[source_name] = ([], [])

            covered, missed = file_lines[source_name]
            for line in source_file.findall("./line"):
                try:
                    line_number = int(line.get("nr"))
                    covered_instructions = int(line.get("ci", "0"))
                except (TypeError, ValueError):
                    continue

                if covered_instructions > 0:
                    covered.append(line_number)
                else:
                    missed.append(line_number)

        if self.use_report_coverage_feature_flag:
            return {
                report_name: self._coverage_result(covered, missed)
                for report_name, (covered, missed) in file_lines.items()
            }

        covered = []
        missed = []
        for report_name, (file_covered, file_missed) in file_lines.items():
            if self._is_requested_file(report_name):
                covered.extend(file_covered)
                missed.extend(file_missed)

        return self._coverage_result(covered, missed)

    @staticmethod
    def _line_numbers(value) -> list:
        if value is None:
            return []

        if isinstance(value, dict):
            value = value.get("lines", value.get("line_numbers", []))

        if not isinstance(value, (list, tuple, set)):
            value = [value]

        numbers = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("line", item.get("line_number", item.get("number")))
            elif isinstance(item, (list, tuple)) and item:
                item = item[0]

            try:
                numbers.append(int(item))
            except (TypeError, ValueError):
                continue
        return numbers

    def parse_json_diff_coverage_report(self) -> Tuple[list, list, float]:
        report_path = self.diff_coverage_report_path or self.file_path
        with open(report_path, "r") as report_file:
            report = json.load(report_file)

        files = report.get("files", report) if isinstance(report, dict) else {}
        if isinstance(files, list):
            converted_files = {}
            for item in files:
                if not isinstance(item, dict):
                    continue
                name = item.get("filename", item.get("file", item.get("name")))
                if name:
                    converted_files[name] = item
            files = converted_files

        selected = None
        if isinstance(files, dict):
            for name, data in files.items():
                if self._is_requested_file(str(name)):
                    selected = data
                    break

        if selected is None:
            return [], [], 0

        if isinstance(selected, list):
            selected = {"violation_lines": selected}
        if not isinstance(selected, dict):
            return [], [], 0

        covered = self._line_numbers(
            selected.get("covered_lines", selected.get("lines_covered"))
        )
        missed = self._line_numbers(
            selected.get(
                "violation_lines",
                selected.get("violations", selected.get("missed_lines")),
            )
        )

        result_covered, result_missed, computed_percentage = self._coverage_result(
            covered, missed
        )

        percentage = selected.get(
            "percent_covered",
            selected.get(
                "coverage_percentage",
                selected.get("coverage", selected.get("percent")),
            ),
        )
        if percentage is None:
            return result_covered, result_missed, computed_percentage

        try:
            percentage = float(percentage)
        except (TypeError, ValueError):
            return result_covered, result_missed, computed_percentage

        if percentage > 1:
            percentage /= 100

        return result_covered, result_missed, percentage