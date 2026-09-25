"""
ML Challenge 2026: Business Entity Resolution
Root Test Runner

Runs all unit and integration tests across the project.
Usage:
    python run_tests.py
"""

import os
import sys
import unittest


def main():
    repo_root = os.path.abspath(os.path.dirname(__file__))
    src_dir = os.path.join(repo_root, "code", "business_entity_resolution", "src")
    tests_dir = os.path.join(repo_root, "tests")

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)

    print(f"Discovering and running tests in: {tests_dir}")
    suite = unittest.defaultTestLoader.discover(tests_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
