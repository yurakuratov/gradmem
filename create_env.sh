conda env create -f conda_env.yaml

conda run -n py311_pt2.6_cu12.4 pip install --upgrade pip
conda run -n py311_pt2.6_cu12.4 pip install "flash-attn==2.7.4.post1" --no-build-isolation --prefer-binary
conda run -n py311_pt2.6_cu12.4 pip install wandb weave