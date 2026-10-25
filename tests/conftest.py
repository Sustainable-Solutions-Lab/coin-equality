"""Pytest hooks for the regression suite: print the per-case worst differences."""

from tests.regression_cases import DIFFERENCE_REPORT


def pytest_terminal_summary(terminalreporter):
    if not DIFFERENCE_REPORT:
        return
    terminalreporter.section('worst relative difference per case')
    for case_name, difference, key in DIFFERENCE_REPORT:
        terminalreporter.write_line(f'{case_name:<20s} {difference:10.2e}  ({key})')
