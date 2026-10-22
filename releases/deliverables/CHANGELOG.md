# Changelog

All notable changes to this project are recorded here. The format
follows Keep a Changelog, and the project uses semantic versioning.

## 0.1.0

### Added

- First release of the deliverable-checker project, extracted from the AI reliability
  lab as a standalone package.
- The project module, the shared core modules it needs, the project's
  evaluation fixtures, and a per project test suite.
- A `demo` command that runs the project on a worked example with no
  model and no network access.
- An `eval` command that runs the project's own evaluation cases and
  reports agreement counts with the label provenance attached.
