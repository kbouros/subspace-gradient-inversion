# Gradient Inversion through Linear and Affine Subspace Constraints

Implementation accompanying **Gradient Inversion through Linear and Affine Subspace Constraints**, a manuscript being prepared for submission to arXiv.

This project studies image-domain gradient inversion using the linear and affine structure of fully connected (FC) layer gradients. It combines cosine gradient matching with soft subspace penalties or a hard subspace parameterization, optionally using constraints from multiple FC layers. It also includes the Cocktail Party Attack (CPA) baseline and a two-stage embedding/feature inversion pipeline for VGG-16.

**See [pipeline_demo.ipynb](pipeline_demo.ipynb)** for example commands, configuration choices, reconstruction visualization, and the percentile-based contrast stretching described in the paper.

The notebook is an illustrative workflow. Running its examples is not equivalent to reproducing every experiment in the paper.

## Methods

| Method | Description |
| --- | --- |
| Soft Linear / Soft Affine | Optimize images or embeddings with penalties for their distance to the estimated subspaces. |
| Hard Linear / Hard Affine | Optimize coefficients in the first attacked layer's estimated subspace, with optional intermediate-layer penalties. |
| IG-style gradient matching | Optimize images using cosine gradient matching and the TV prior. |
| Cocktail Party Attack (CPA) | Recover images or embeddings using the independent-component-analysis approach of Kariyappa et al. |

For an FC layer, aggregate gradients satisfy `G_W = Delta @ X` and `g_b = Delta @ 1`. Dividing a weight-gradient row by its corresponding nonzero bias-gradient entry gives an affine combination of the layer inputs. The affine estimator uses bias-normalized rows and a reliability-weighted, centered SVD. The linear estimator uses an uncentered SVD. Note that no affine hull is estimated if there aren't enough nonzero bias-gradient entries.

## Datasets and models

| Dataset | Supported model | Preparation |
| --- | --- | --- |
| CIFAR-10 / CIFAR-100 | `fc<L>`, with `L >= 2` | Downloaded by the training pipeline through torchvision. |
| Tiny-ImageNet | `fc<L>`, with `L >= 2` | Download separately; the notebook includes validation-folder preparation. |
| ImageNet | Pretrained `vgg16` | Obtain the dataset separately and prepare validation class folders. The notebook shows a Colab/Drive example. |

FC models support `--act=relu` and `--act=leaky_relu`. Use the same dataset, model, hidden width, activation, and checkpoint for training, collection, and inversion. VGG uses pretrained weights; skip the FC training step for that configuration.

## Example FC workflow

The following illustrates the CIFAR-10 FC-3 pipeline and the noiseless hyperparameters in manuscript Table 7. 

```sh
python src/train.py --ds=cifar10 --model=fc3 --h_dim=512 --act=relu --epochs=20 --batch_size=128 --dry_run
python src/collect_grads.py --ds=cifar10 --model=fc3 --h_dim=512 --act=relu --defense=nodef --n_batch=5
python src/attack.py --ds=cifar10 --model=fc3 --act=relu --h_dim=512 --attack=gm --subspace_type=affine --sub_input=0.0001 --sub_intermediate=0.0001 --tv=0.0001 --lr=0.05 --batch_size=32 --n_batch=5 --n_iter=25000 --n_restarts=2 --dry_run
```

Training, collection, and attack must use the same architecture, activation, width, and model checkpoint.

To compare the FC methods, use these changes to the attack command:

| Variant | Options |
| --- | --- |
| Soft Linear | `--subspace_type=linear --sub_input=0.0001 --sub_intermediate=0.0001` |
| Hard Affine | `--subspace_type=affine --hard_sub --sub_input=0 --sub_intermediate=0.0001` |
| Hard Linear | `--subspace_type=linear --hard_sub --sub_input=0 --sub_intermediate=0.0001` |
| IG | `--attack=gm --sub_input=0 --sub_intermediate=0 --tv=0.001 --lr=0.05` |
| CPA | `--attack=cp --sub_input=0 --sub_intermediate=0 --lr=0.001 --tv=3.1 --decor=1.47 --T=12.4 --n_restarts=1` |

Replace options instead of appending contradictory flags. `--hard_sub` is used only for the hard variants. Set `--act=leaky_relu` consistently across the three pipeline stages for that activation. For the manuscript's noisy FC variants, reduce the applicable subspace weights to `0.00001`.

## Important configuration options

| Option | Meaning |
| --- | --- |
| `--batch_size` | True batch size of the collected gradients; selects the gradient file. |
| `--batch_size_attack` | Number of reconstructed samples and requested subspace dimension parameter. Defaults to the true batch size. |
| `--hard_sub` | Use a hard subspace parameterization for the first attacked layer's inputs. |
| `--sub_input` | Weight of the soft subspace penalty on inputs or reconstructed embeddings. |
| `--sub_intermediate` | Weight of subspace penalties at intermediate FC layers. |
| `--subspace_type` | `linear` or `affine`. |
| `--use_labels` | Assume the ground-truth labels are known. Use the correct attack batch size in this setting. |
| `--n_restarts` | Number of GM initializations. CPA uses its deterministic initialization. |

## VGG embedding inversion

For a noiseless, unknown-label Hard Affine example corresponding to Table 8:

```sh
python src/collect_grads.py --ds=imagenet --model=vgg16 --defense=nodef --n_batch=5
python src/attack.py --ds=imagenet --model=vgg16 --attack=gm --subspace_type=affine --hard_sub --sub_intermediate=0.0001 --lr=0.06 --tv=0.1 --nv=0.1 --l1=0.001 --decor=2 --batch_size=32 --n_batch=5 --n_iter=25000 --n_restarts=2 --n_iter_fi=0 --dry_run
```

`--n_iter_fi=0` evaluates embeddings only. A positive value enables feature inversion. `--fi_method=gm` combines feature and gradient matching, while `--fi_method=direct` uses feature matching alone.

## Perturbed gradients

An example of collecting clipped gradients with Gaussian noise:

```sh
python src/collect_grads.py --ds=cifar10 --model=fc3 --act=relu --h_dim=512 --defense=dp --C=1.0 --sigma=0.0046 --n_batch=5
```

Use the same `--defense=dp --C=1 --sigma=0.0046` when attacking. At batch size 32, this noise multiplier corresponds to approximately 0 dB SNR for the FC-3 configuration evaluated in Table 6. The achieved SNR depends on the model and data.

## Reconstructions and contrast stretching

The notebook loads saved reconstructions, compares methods, and exports figures as PNG or PDF. Its `percentile_stretch` helper implements the paper's optional post-processing:

1. Compute the 0.1st and 99.9th percentiles over the **entire reconstructed batch**.
2. Apply the same intensity transformation to every image in that batch.
3. Clip the result to `[0, 1]`.

This is a post-processing operation; it does not change the inversion optimization or observed gradients. The paper applies it to the selected noisy FC subspace reconstructions. A `+` suffix identifies a stretched result.

In `create_reconstruction_comparisons`, `stretch_algorithms=[2]` applies stretching to the third reconstruction file in `pkl_paths`. No stretching is applied with `stretch_algorithms=None`. Plotting a stretched reconstruction does not automatically recompute the saved numerical metrics. For before/after comparisons, show both versions and recompute metrics if reporting quantitative improvements.

## Code provenance and acknowledgements

The experiment framework is adapted from the official [Cocktail Party Attack repository](https://github.com/facebookresearch/cocktail_party_attack), accompanying:

Sanjay Kariyappa, Chuan Guo, Kiwan Maeng, Wenjie Xiong, G. Edward Suh, Moinuddin K Qureshi, and Hsien-Hsin S. Lee. **Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning Using Independent Component Analysis.** ICML 2023, PMLR 202:15884-15899. [Paper](https://proceedings.mlr.press/v202/kariyappa23a.html).

Inherited components include CPA, the training and gradient-collection framework, feature inversion, model wrappers, and experiment utilities. This adaptation adds linear/affine subspace estimation, soft and hard constraints, intermediate-layer penalties, Hit-and-Run initialization, variable-depth FC models, Hungarian batch matching, and support for batch-size uncertainty. `dataset_utils.py` is adapted from CPA's `datasets.py`.

The cosine gradient-matching formulation follows Geiping et al., **Inverting Gradients - How easy is it to break privacy in federated learning?**, NeurIPS 2020. The label "IG" refers to the implemented gradient-matching baseline and does not imply use of that paper's original code unchanged.

CPA acknowledges MIT-licensed [GradAttack](https://github.com/Princeton-SysML/GradAttack), and the modified VGG code contains torchvision-derived source. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/` for retained notices. This project is an independent adaptation; attribution does not imply upstream endorsement.

## License

Except for separately identified third-party material, this code and its accompanying repository documentation are distributed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**; see [LICENSE](LICENSE). The CPA-derived material remains subject to its original license and copyright notices. The same license is selected here for this adaptation's contributions. Separately licensed portions retain their original terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/`.
