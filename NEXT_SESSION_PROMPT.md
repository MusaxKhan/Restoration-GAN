Copy everything below the line into a new Claude session opened in `E:\University\Semester 7\GenAI\Assignment 1`.

---

You are continuing my university Generative-AI Assignment 1 (repo https://github.com/MusaxKhan/Restoration-GAN, local folder `E:\University\Semester 7\GenAI\Assignment 1`). Training, evaluation, the app, the Docker setup and the IEEE report are DONE and pushed. Start by reading `HANDOFF.md` in the repo root (full state, constraints, gotchas), then `README.md`, then run `git status` and `git log --oneline | head`.

Rules: commits must be authored only by me (no Co-Authored-By trailer); use real measured numbers only; install nothing on C: (use E:); ask me before downloads, OAuth/ToS, public sharing or installs; keep your messages short - I'm tired. The deadline is tonight at midnight (check the date/time).

What is left, in order (details in HANDOFF.md section 4):
1. I create a GitHub Release `v1.0` containing `data/models_final.zip`; then you test `python scripts/download_models.py` against the real URL.
2. Install Docker Desktop on E: (ask me first, it needs admin and probably a reboot), then run `docker compose up --build` once and verify all four workspaces at http://localhost:3000 (and the optional MLflow UI via `docker compose --profile tracking up mlflow`).
3. Give me a short shot list for the 5-7 minute demo video (I record and upload to YouTube).
4. After I give you the YouTube link: put it into `report/main.tex` (replace `YOUTUBE_LINK_TO_BE_ADDED`), recompile with Tectonic (command in HANDOFF.md), check the PDF, commit and push, and tell me exactly what to upload to Google Classroom.
5. Final repo check (README accuracy, clean `git status`, last push).

Begin with step 1: tell me exactly what to click on GitHub, and wait for my "published".
