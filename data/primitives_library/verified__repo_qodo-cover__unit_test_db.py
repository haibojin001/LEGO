import argparse
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import load_only, scoped_session, sessionmaker

from cover_agent.report_generator import ReportGenerator


Base = declarative_base()


class UnitTestGenerationAttempt(Base):
    __tablename__ = "unit_test_generation_attempts"

    id = Column(Integer, primary_key=True)
    run_time = Column(DateTime, default=datetime.now)
    status = Column(String)
    reason = Column(Text)
    exit_code = Column(Integer)
    stderr = Column(Text)
    stdout = Column(Text)
    test_code = Column(Text)
    imports = Column(Text)
    language = Column(String)
    prompt = Column(Text)
    source_file = Column(Text)
    original_test_file = Column(Text)
    processed_test_file = Column(Text)


class UnitTestDB:
    def __init__(self, db_connection_string):
        self.engine = create_engine(db_connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def insert_attempt(self, test_result: dict):
        with self.Session() as session:
            new_attempt = UnitTestGenerationAttempt(
                run_time=datetime.now(),
                status=test_result.get("status"),
                reason=test_result.get("reason"),
                exit_code=test_result.get("exit_code"),
                stderr=test_result.get("stderr"),
                stdout=test_result.get("stdout"),
                test_code=test_result.get("test", {}).get("test_code", ""),
                imports=test_result.get("test", {}).get("new_imports_code", ""),
                language=test_result.get("language"),
                prompt=test_result.get("prompt"),
                source_file=test_result.get("source_file"),
                original_test_file=test_result.get("original_test_file"),
                processed_test_file=test_result.get("processed_test_file"),
            )
            session.add(new_attempt)
            session.commit()
            return new_attempt.id

    def get_all_attempts(self):
        with self.Session() as session:
            attempts = session.query(UnitTestGenerationAttempt).all()

        return [
            {
                "id": attempt.id,
                "status": attempt.status,
                "reason": attempt.reason,
                "exit_code": attempt.exit_code,
                "stderr": attempt.stderr or "",
                "stdout": attempt.stdout or "",
                "test_code": attempt.test_code or "",
                "imports": attempt.imports or "",
                "language": attempt.language,
                "prompt": attempt.prompt,
                "source_file": attempt.source_file,
                "original_test_file": attempt.original_test_file,
                "processed_test_file": attempt.processed_test_file,
            }
            for attempt in attempts
        ]

    def dump_to_report(self, report_filepath):
        ReportGenerator.generate_report(self.get_all_attempts(), report_filepath)


def dump_to_report(
    path_to_db="cover_agent_unit_test_runs.db",
    report_filepath="test_results.html",
):
    unittest_db = UnitTestDB(f"sqlite:///{path_to_db}")
    unittest_db.dump_to_report(report_filepath)


def dump_to_report_cli():
    parser = argparse.ArgumentParser(description="Generate a unit test report.")
    parser.add_argument(
        "--path-to-db",
        type=str,
        default="cover_agent_unit_test_runs.db",
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--report-filepath",
        type=str,
        default="test_results.html",
        help="Path to the HTML report file.",
    )
    args = parser.parse_args()
    dump_to_report(
        path_to_db=args.path_to_db,
        report_filepath=args.report_filepath,
    )