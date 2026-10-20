NEURO ATLAS - 14 teaching examples

Requires Python 3.11 + PyTorch. DQN/PPO require gymnasium and stable-baselines3.
Each file is standalone: python NAME.py --steps 20 --device auto
--steps means parameter updates except DQN/PPO, where it means requested environment transitions. PPO rounds to full rollout lengths.
Examples deliberately simplify the papers. Read each course page and file comments for omissions.
Most use synthetic data; CTC is not a speech recognizer; DDPM has no reverse sampler; wav2vec2_core omits quantization and diversity loss.
No pretrained weights or datasets are downloaded. Finite loss / successful execution is not evidence of learning quality.
