# Third-party source attribution

ODIE's original code is licensed under the [MIT License](../../LICENSE). This grant does not relicense third-party code or upstream-derived portions of patches, which retain their applicable terms. Source images and figure excerpts also retain their original terms; see [data attribution](../../data/README.md).

| Dependency | Pinned revision | Source | License notice |
|---|---|---|---|
| siren | `4df34baee3f0f9c8f351630992c1fe1f69114b5f` | https://github.com/vsitzmann/siren.git | [MIT](siren/LICENSE) |
| finer | `07a9a922e5bc20323d080ba382f71d333e9c7482` | https://github.com/liuzhen0212/FINER.git | No root LICENSE/COPYING/NOTICE at this revision; no license is inferred. |
| miner | `f8f6ed442048f883cf70541117f0754ad437cc69` | https://github.com/vishwa91/MINER.git | [MIT](miner/LICENSE) |
| ingp | `abe236ee00cf90cfca6e36e65c00435d5b21f50a` | https://github.com/NVlabs/instant-ngp.git | [NVIDIA Source Code License](ingp/LICENSE.txt); includes a non-commercial research/evaluation use restriction. |
| neural_experts | `5ff5b59184db6cdeb2f7902e18786380c8652ba2` | https://github.com/sitzikbs/Neural-Experts.git | No root LICENSE/COPYING/NOTICE at this revision; no license is inferred. |

tiny-cuda-nn and the other INGP submodules are fetched at the revisions in [upstreams.lock.json](../../upstreams.lock.json); retain the notices delivered with each checkout. Optional JBIG support uses the system JBIG-KIT library under its separate terms. Third-party sources are not bundled as a modified vendor tree.
