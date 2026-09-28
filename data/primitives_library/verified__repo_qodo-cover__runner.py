import subprocess
import time


class Runner:
    @staticmethod
    def run_command(command: str, max_run_time_sec: int, cwd: str = None):
        command_start_time = int(time.time() * 1000)
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=max_run_time_sec,
            )
            return result.stdout, result.stderr, result.returncode, command_start_time
        except subprocess.TimeoutExpired:
            return "", "Command timed out", -1, command_start_time