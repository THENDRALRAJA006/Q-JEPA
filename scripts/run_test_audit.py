import pytest
import json

class CountPlugin:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
            elif report.skipped:
                self.skipped += 1

def main():
    plugin = CountPlugin()
    exit_code = pytest.main(["tests/", "-q", "--disable-warnings"], plugins=[plugin])
    res = {
        "exit_code": int(exit_code),
        "passed": plugin.passed,
        "failed": plugin.failed,
        "skipped": plugin.skipped,
        "total": plugin.passed + plugin.failed + plugin.skipped
    }
    with open("results/metrics/test_suite_execution_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print("TEST_SUITE_SUMMARY:", json.dumps(res))

if __name__ == "__main__":
    main()
