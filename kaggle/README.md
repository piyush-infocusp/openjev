# Kaggle GPU training

One-time setup (you, not the agent — needs your credentials):

1. `pip install kaggle`
2. Create an API token on kaggle.com (Account → Create New Token) and save it
   to `~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`.
3. Verify your phone number on Kaggle (Account → Phone Verification) — GPU
   and internet access on kernels require this.
4. `export KAGGLE_USERNAME=<your kaggle username>`

Command sequence:

```bash
bash kaggle/run.sh bundle          # build build/kaggle_dataset/ (source zip)
bash kaggle/run.sh upload          # kaggle datasets create/version
bash kaggle/run.sh push            # render kernel-metadata.json, push kernel
bash kaggle/run.sh status          # check current status
bash kaggle/run.sh wait            # poll every 60s until complete/error
bash kaggle/run.sh pull            # pull output into runs/<timestamp>/
```

Use `--dry-run` before the subcommand (e.g. `bash kaggle/run.sh --dry-run push`)
to print the exact commands without touching Kaggle.

To change training args, pass a custom args file to `make_bundle.sh` (or
`kaggle/run.sh bundle`'s underlying script) before pushing, e.g.
`kaggle/make_bundle.sh my_run_args.txt`.

Check the weekly GPU quota at https://www.kaggle.com/docs/efficient-gpu-usage
before pushing a kernel — GPU sessions are capped per week per account.
