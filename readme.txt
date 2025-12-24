download z-image-turbo model

Envs perpare:
pip install git+https://github.com/huggingface/diffusers
pip install transformers

run python run.py

Details:
enable vae tiling function through interface : pipe.vae.enable_tiling()

vae tiling reference code path in envs:
C:\Users\admin\miniforge3\envs\z-image\Lib\site-packages\diffusers\pipelines\z_image\pipeline_z_image.py -> self.vae.decode
C:\Users\admin\miniforge3\envs\z-image\Lib\site-packages\diffusers\models\autoencoders -> def tiled_decode

