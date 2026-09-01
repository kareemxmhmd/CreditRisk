# Contributing to CreditRisk Engine

Welcome to the CreditRisk Engine project! We appreciate your interest in contributing to our AI-powered loan decisioning system.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd CreditRisk
   ```

2. **Install dependencies:**
   We recommend using a virtual environment (e.g. `venv` or `conda`).
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the tests:**
   Before making changes, verify that the current pipeline and tests run successfully.
   ```bash
   pytest tests/ -v
   python run_all.py
   ```

## Development Guidelines

- **Code Quality:**
  - Please ensure all public functions and methods use appropriate type hints.
  - Write concise docstrings for all modules, classes, and significant functions.
  - Follow standard Python PEP8 formatting.
  - Avoid hardcoding magic numbers or paths. Put configuration variables in `src/config.py`.

- **Logging:**
  - Use the built-in `logging` module. Do not use `print()` in production code.

- **Testing:**
  - Write unit tests for new features inside the `tests/` folder.
  - Make sure your tests cover both positive and negative cases.

## Submitting a Pull Request

1. Create a new branch from `main`.
2. Commit your changes with clear, descriptive commit messages.
3. Push your branch and open a Pull Request against `main`.
4. Our CI workflow will automatically run `pytest` and a linter (flake8/ruff) on your PR. Ensure that the CI passes successfully before requesting a review.
