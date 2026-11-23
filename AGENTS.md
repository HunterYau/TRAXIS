# Runpod account controls

- The user permits connecting to Runpod and viewing account and pod information.
- Get the user's explicit approval before creating, starting, or resuming any pod, including through deployment commands or templates.
- Account setup alone does not authorize starting compute or simulations.
- Never print credentials or commit them to this repository.

# Hardware-aware implementation

- The user's local computer has an NVIDIA GeForce RTX 3060 Ti with 8 GB dedicated GPU memory and approximately 16 GB system RAM.
- The user also has an existing Runpod H200 NVL pod (`hetsv718t16liy`), previously verified with about 141 GiB GPU memory and 283 GB host RAM. Consider this cloud option for workloads that exceed local resources, while choosing efficient algorithms for both machines. Recheck its current allocation, availability, state, and price before relying on it; a stopped pod does not reserve its GPU. The explicit approval requirements above still apply.
- Account for this hardware when choosing algorithms and preparing local runs. Estimate peak RAM, GPU memory, and runtime before large calculations; check currently available memory before launch.
- Prefer memory-efficient methods such as matrix-free operations, batching, or streaming when suitable. Preserve accuracy rather than silently reducing mesh resolution, time resolution, precision, or validation requirements to fit memory.
- Clearly distinguish local hardware requirements from cloud requirements. Do not assume a local run has H100/H200 resources, and do not claim a configuration fits solely from an unverified estimate.
