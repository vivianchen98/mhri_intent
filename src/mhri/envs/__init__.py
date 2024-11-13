from gymnasium.envs.registration import register

register(
    id="GAN-v1",
    entry_point="mhri.envs.gnomes_at_night:GnomesAtNightEnv",
)
