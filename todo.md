## todo

python logger

##

Training

Input:

- Ligand -> EGNN (Encoder) -> Diffusion -> EGGN (Decoder) -> Ligand`

jetzt grade

- Protein -> ESM2 -> Vector (Context)

man könnte

- Protein -> EGNN-Protein (Encoder) -> (selbe output wie Ligand Encoder)

## Plan

1. ESM pocket in diffusion
   1. VAE only ligands (wenn fertig trainiert) or check if can use pretrained VAE / Diffusion
   2. freeze VAE and put ESM pocket into Diffusion (sanity check)
      1. hoffe loss geht runter und docking Score geht hoch
   3. Expectation check quality of Docking Score

Next steps: 2. ESM pocket diffusion and VAE (Encoder and Decoder) -> hier müsste ich VAE neu trainiere 3. EGNN Protein Encoder in Diffusion -> hier müsste ich Diffusion neu trainieren

Inference
