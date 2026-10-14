import os,sys
from pathlib import Path
root=Path(__file__).resolve().parent;repo=root/'repo'
import native_aime_formal as n
assert os.environ.get('RPAS_MAAS_EMBEDDING_PATH'), 'Set local embedding path'
os.environ['CUDA_VISIBLE_DEVICES']=''
m=root/'maas_import_probe'
if not m.exists():n.copy_tree(repo/'external_baselines/MaAS',m)
n.patch_maas_optional_provider_import(m);n.patch_maas_provider_surface(m);n.patch_maas_optional_backoff(m);n.patch_maas_optional_tools(m);n.patch_maas_embedding_path(m);n.make_maas_config(m,os.environ.get('RPAS_ENDPOINT','http://127.0.0.1:8000/v1'),6144)
from maas_context_guard import patch_template_directory
for kind in ['train','test']:patch_template_directory(m/f'maas/ext/maas/scripts/optimized/MATH/{kind}/template',concise=True)
os.chdir(m);sys.path.insert(0,str(m));os.environ['METAGPT_PROJECT_ROOT']=str(m)
from maas.configs.models_config import ModelsConfig
from maas.ext.maas.scripts.optimizer import Optimizer
model=ModelsConfig.default().get('qwen35_9b')
assert model.model=='Qwen/Qwen3.5-9B'
print('MAAS_NATIVE_IMPORT_PASS',model.temperature)
