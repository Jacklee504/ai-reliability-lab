# Cleanup

Two things need a human, because the safety hooks in the working
environment block the file and index operations. Both are documented
here so they are not lost.

## 1. Remove the build artefacts from the released packages

An earlier version of the install check built the wheel inside the
release directory, which left a `build` directory and an egg-info
directory in each of the thirteen committed packages. The check now
builds in a throwaway copy, so they will not come back, but the
existing ones are still tracked by git.

The commands to run, from the repository root:

1. Remove the `build` directories under `releases/`, and the
   `.egg-info` directories under `releases/*/src/`, from the working
   tree.
2. Untrack the same paths with git, so the index no longer holds
   them.
3. Commit the result.

Until that is done, one test fails:

```
tests/test_release.py::test_a_release_carries_no_build_artifacts
```

The test is saying exactly this, so it will pass once the directories
are gone. The paths are already listed in `.gitignore`, so nothing
like them will be added again.

## 2. Push the commits

The Git credential manager cannot open an interactive prompt from a
non-interactive session, so the push has to be run in a terminal
where it can ask for your credentials.

```bash
cd /Users/jacklee/Other/Me/Projects/Misc/ai-reliability-lab
git push origin main
```

If the credential manager still cannot prompt, point the remote at a
personal access token instead:

```bash
git remote set-url origin https://<token>@github.com/JackLee504/ai-reliability-lab.git
git push origin main
```

## 3. Optional: remove the earlier hand written extraction

`claim-auditor/` is the earlier hand-written extraction of the claims
project. `releases/claims/` is generated from the lab and is the
canonical one, and the two will drift because only one of them is
rebuilt. The hand-written directory can go.

The tests it contained that the lab did not have were moved into the
lab's own per project test files, so nothing is lost by removing it.
