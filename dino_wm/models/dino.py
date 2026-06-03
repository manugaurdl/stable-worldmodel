import sys
import importlib.abc
import importlib.machinery
import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True


def _enable_py39_dinov2():
    """Make the (latest) facebookresearch/dinov2 hub code importable on Python 3.9.

    torch.hub.load pulls dinov2 `main`, whose layers/{block,attention}.py use PEP 604
    `X | None` annotations that are *evaluated* at function-definition time and so raise
    TypeError on Python < 3.10 (the pinned dino_wm conda env is 3.9). We inject
    `from __future__ import annotations` into dinov2 modules *in memory* as they import
    -- the on-disk torch hub cache is never modified, no .pyc is written, and runtime
    behaviour (hence the features) is unchanged. No-op on Python >= 3.10.
    """
    if sys.version_info >= (3, 10):
        return
    _FUTURE = "from __future__ import annotations\n"

    class _Loader(importlib.machinery.SourceFileLoader):
        def get_source(self, fullname):
            src = super().get_source(fullname)
            return src if _FUTURE in src else _FUTURE + src

        def get_code(self, fullname):  # always compile from (patched) source, ignore .pyc
            return compile(self.get_source(fullname), self.path, "exec", dont_inherit=True)

        def set_data(self, *a, **k):  # never write .pyc into the hub cache tree
            return None

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] != "dinov2":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or not spec.origin or not spec.origin.endswith(".py"):
                return None
            spec.loader = _Loader(name, spec.origin)
            return spec

    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        _enable_py39_dinov2()
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb