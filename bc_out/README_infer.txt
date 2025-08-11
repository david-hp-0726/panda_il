Inference steps:
1) obs_n = (obs - obs_mean) / obs_std
2) a_scaled = model(obs_n)
3) a = a_scaled * act_scale   # Δq in joint space
