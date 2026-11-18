# TacInno-VLA on pi0.5 (PaliGemma 2B prefix + Gemma 300M action expert); see tacinno/models/policy/tacinno_vla.py.
model = dict(
    type="TacInnoVLA",
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    action_horizon=32,              # 32-step action chunk of the 120-D unified action
    max_token_len=150,
    proprio_history=0,              # proprio tokens (joints of t-7 .. t); 8 in the Lift Can recipe, 0 elsewhere
    use_tactile=True,
    tactile_history=(8, 4),         # tactile observations of frames t-8, t-4 (and t)
    tactile_tokens_per_frame=10,    # per gel and frame: summary token + 3 x 3 pooled patch tokens
    use_code_queries=True,          # 4 learned code queries read out by the code head
    code_head=True,
    code_to_action=False,           # no other token attends to the code queries
    expert_updates=("tactile", "code"),
    gate_init=0.1,
    code_loss_weight=0.119,         # lambda_code
    contact_loss_weight=0.467,      # lambda_contact
)
