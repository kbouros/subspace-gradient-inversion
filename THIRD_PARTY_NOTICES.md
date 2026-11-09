# Third-party notices and modifications

## Cocktail Party Attack (CPA)

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

Source: https://github.com/facebookresearch/cocktail_party_attack

License: Creative Commons Attribution-NonCommercial 4.0 International.
Full unmodified upstream text is retained in [LICENSE](LICENSE):
https://creativecommons.org/licenses/by-nc/4.0/

The original research is by Sanjay Kariyappa, Chuan Guo, Kiwan Maeng,
Wenjie Xiong, G. Edward Suh, Moinuddin K Qureshi, and Hsien-Hsin S. Lee.
See the full paper citation in [README.md](README.md).

The following local files are adapted from CPA's correspondingly named
`src/` modules: `attack.py`, `collect_grads.py`, `eval_utils.py`, 
`feature_inversion.py`, `gradient_inversion.py`, `gradient_inversion_utils.py`,
`log_utils.py`, `models.py`, `train.py`, `utils.py`, and `vgg16.py`.
Local `dataset_utils.py` is adapted from upstream `src/datasets.py`.

Modifications include module renaming, variable-depth FC architectures,
activation selection, linear/affine hull estimation, hard and soft 
input/embedding constraints, intermediate-layer regularization,
Hit-and-Run initialization in `initialize_subspace_utils.py`,
rectangular Hungarian assignment for batch-size uncertainty, gradient
clipping/noise utilities, and corresponding command-line, logging, and
checkpoint changes.

These changes are part of an independent adaptation. Upstream names do not
imply sponsorship or endorsement. The new contributions and repository
documentation use CC BY-NC 4.0, except for the separately licensed materials
below. Original authors retain their rights; the adaptation does not claim
ownership of their code.

## GradAttack

CPA's README acknowledges that portions of its project originate from
[Princeton-SysML/GradAttack](https://github.com/Princeton-SysML/GradAttack)
under the MIT License. Its notice is retained in
[LICENSES/GradAttack-MIT.txt](LICENSES/GradAttack-MIT.txt):

Copyright (c) 2021 Princeton-SysML

The upstream CPA acknowledgement does not map each inherited portion to
individual lines. Retaining this notice preserves that upstream exception;
it does not claim the entire CPA-derived repository is MIT-licensed.

## torchvision

The modified `vgg16.py` retains torchvision VGG implementation structure,
configuration tables, model-weight metadata, and helper code, via CPA.
The torchvision BSD 3-Clause license is retained in
[LICENSES/torchvision-BSD-3-Clause.txt](LICENSES/torchvision-BSD-3-Clause.txt).

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Source: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
License: https://github.com/pytorch/vision/blob/main/LICENSE

The source also retains the upstream reference to the SSD port for the
optional VGG feature weights. No weight file is bundled in this release.

## External dependencies, datasets, and manuscript

Installed libraries retain their respective licenses. This notice is not a
complete transitive dependency inventory. Dataset downloads, pretrained
weights, and LPIPS backbones are obtained separately and remain subject to
their applicable terms. The code license does not grant rights to private
data or relabel the manuscript's licensing.
