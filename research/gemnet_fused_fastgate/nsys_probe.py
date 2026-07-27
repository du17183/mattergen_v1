from __future__ import annotations

import torch

from research.gemnet_fused_fastgate.harness import (
    build_c0_generator,
    build_sampler,
    configure_determinism,
    find_gemnet,
    load_states,
    prepare_joint_states,
    run_joint_score,
)


def main() -> int:
    configure_determinism()
    generator = build_c0_generator(sampling_steps=1000)
    sampler = build_sampler(generator)
    diffusion_module = generator.model.diffusion_module
    gemnet = find_gemnet(generator.model)
    device = next(gemnet.parameters()).device
    states = prepare_joint_states(load_states(4), sampler, device)
    with torch.inference_mode():
        for index in range(10):
            run_joint_score(diffusion_module, sampler, states[index % len(states)])
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for index in range(8):
            torch.cuda.nvtx.range_push("C0_B1_JOINT_CFG_FORWARD")
            run_joint_score(diffusion_module, sampler, states[index % len(states)])
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
