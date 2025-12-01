To download the data just run
```bash
uv run python -m scripts.data_processing.exorl.download --folder <your folder>
```
a folder `original` will be created inside the provided folder.

To replay the data using the newest version of mujoco
```bash
uv run python -m scripts.data_processing.exorl.update_all --folder <your folder>
```
new datasets will be saved in the folder `processed`.
