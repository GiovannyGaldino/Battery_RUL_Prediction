# Public release checklist

Before pushing to GitHub:

1. Run `git status` and confirm that no data files are staged.
2. Check for large files with a short Python script or PowerShell equivalent.
3. Confirm there are no machine-specific absolute paths in Python or Markdown files.

4. Run a syntax check:

```bash
python -m compileall -q src scripts tools
```

5. Commit only code, documentation, and configuration templates.
