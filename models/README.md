# MediaPipe models

The binary model assets are not committed to Git. Run `setup.bat`, or run this
from the repository root:

```bash
python download_models.py
```

The downloader fetches the official MediaPipe model files listed in
`download_models.py`, writes each to a `.part` file, verifies its exact size and
SHA-256, and only then moves it into this directory. Valid existing files are
left untouched.
