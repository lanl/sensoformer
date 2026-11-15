import builtins
import importlib
import sys


def test_public_import():
    from sensoformer import SensOFormer

    assert SensOFormer.__name__ == "SensOFormer"


def test_script_imports_have_no_output_side_effects_or_wandb_dependency(
    monkeypatch, tmp_path
):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("wandb was imported while logging was disabled")
        return original_import(name, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("scripts.pretrain", None)
    sys.modules.pop("scripts.inference", None)
    importlib.import_module("scripts.pretrain")
    importlib.import_module("scripts.inference")

    assert list(tmp_path.iterdir()) == []
