# Contributing

Thank you for your interest in contributing to RolloTree.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install in development mode:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Run the tests to make sure everything works:

   ```bash
   pytest
   ```

## Making Changes

1. Create a feature branch from `main`.
2. Make your changes and add tests if applicable.
3. Run `pytest` and ensure all tests pass.
4. Commit with a clear message describing the change.
5. Open a pull request against `main`.

## Reporting Issues

Open a GitHub issue at https://github.com/koftezz/rolling-lookahead-dt/issues with:
- A description of the problem or feature request.
- Steps to reproduce (for bugs).
- Python version and OS.

## Code Style

- Follow existing code conventions in the project.
- Use type hints where they exist.
- Keep functions focused and well-documented.

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0-or-later license.
