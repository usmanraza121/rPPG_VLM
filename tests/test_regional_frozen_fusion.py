"""Frozen-candidate invariants and real trainer/checkpoint/audit integration."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from reproducibility import initialize_random_state
from neural_methods.model.regional_frozen_fusion import candidate_summaries
from neural_methods.trainer.RegionalDINOTrainer import (
    RegionalDINOTrainer, build_model, build_loss, initialize_frozen_source,
    model_arguments, validate_model_checkpoint,
)
from tools.audit_regional_dino import main as audit_main

ROOT = Path(__file__).resolve().parents[1]


def config_for(root, mode='waveform', source=None):
    with contextlib.redirect_stdout(io.StringIO()):
        c = get_config(argparse.Namespace(config_file=str(ROOT/'configs/train_configs/ablations'/f'regional_frozen_fusion_{mode}.yaml')))
    c.defrost()
    c.DEVICE = 'cpu'
    c.MODEL.REGIONAL_ST.HIDDEN_DIM = 8
    c.MODEL.REGIONAL_ST.CONTEXT_DIM = 4
    c.MODEL.REGIONAL_ST.GRID_SIZE = 2
    c.TRAIN.EPOCHS = 3
    c.TRAIN.REGIONAL_DINO.CHECKPOINT_START_EPOCH = 1
    c.TRAIN.FROZEN_FUSION.SOURCE_CHECKPOINT = str(source or root/'source.pth')
    c.TRAIN.FROZEN_FUSION.SOURCE_SHA256 = ''
    c.MODEL.MODEL_DIR = str(root/'checkpoints')
    c.LOG.PATH = str(root/'run')
    c.TEST.METRICS = ['MAE']
    c.freeze()
    return c


def samples():
    t = torch.arange(64)/30
    y = torch.stack([torch.sin(2*torch.pi*f*t) for f in [1., 1.2, 1.4, 1.6]])
    x = 110+torch.rand(4, 1, 3, 16, 16)*20 + 2*y[:, :, None, None, None]
    return x, y, [(x[i], y[i], f'recording{i//2}', i % 2) for i in range(4)]


class FrozenFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_candidates_matched_initialization_and_gate_only_update(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            initialize_random_state(100)
            moments = build_model(config_for(root, 'moments'))
            initialize_random_state(100)
            cfg = config_for(root)
            waveform = build_model(cfg)
            x, y, _ = samples()
            moments.train(); waveform.train()
            self.assertFalse(waveform.extractor.training)
            self.assertEqual(sum(p.numel() for p in moments.parameters() if p.requires_grad),
                             sum(p.numel() for p in waveform.parameters() if p.requires_grad))
            a, b = moments(x, return_aux=True), waveform(x, return_aux=True)
            torch.testing.assert_close(a['waveform'], b['waveform'], rtol=0, atol=0)
            torch.testing.assert_close(b['region_weights'], torch.full_like(b['region_weights'], .25))
            torch.testing.assert_close(waveform._moments(x), waveform.extractor._local_features(x/255)[2])
            initial = {k:v.clone() for k,v in waveform.extractor.state_dict().items()}
            old_gate = {k:v.clone() for k,v in waveform.quality_head.state_dict().items()}
            optim = torch.optim.AdamW([p for p in waveform.parameters() if p.requires_grad], lr=.01)
            criterion = build_loss(cfg)
            for _ in range(3):
                optim.zero_grad()
                criterion(waveform(x, return_aux=True), y)['total'].backward()
                optim.step()
            after = waveform(x, return_aux=True)
            for key,value in waveform.extractor.state_dict().items():
                torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in waveform.extractor.parameters()))
            self.assertTrue(any(not torch.equal(v,old_gate[k]) for k,v in waveform.quality_head.state_dict().items()))
            for output in [b, after]:
                torch.testing.assert_close((output['regional_waveforms']*output['region_weights']).sum(1),output['waveform'])
            def centered(v): return v-v.mean(-1,keepdim=True)
            torch.testing.assert_close(centered(b['regional_waveforms']), centered(after['regional_waveforms']), atol=1e-6, rtol=1e-5)
            # Masked pixels cannot affect valid candidates or summary agreements.
            mask = torch.tensor([[True,False,True,True]]).expand(4,-1)
            masked = waveform(x,return_aux=True,region_mask=mask)
            changed=x.clone();changed[..., :8, 8:]=250
            other=waveform(changed,return_aux=True,region_mask=mask)
            torch.testing.assert_close(masked['waveform'],other['waveform'])
            self.assertEqual(float(masked['region_weights'][:,1].abs().max()),0)

    def test_summaries_finite_masked_batch_isolated_and_reference_free(self):
        t=torch.arange(160)/30
        q=torch.stack([torch.sin(2*torch.pi*f*t) for f in [1.,1.,1.8]]).unsqueeze(0)
        valid=torch.tensor([[True,True,False]])
        a=candidate_summaries(q,valid)
        changed=q.clone();changed[:,2]=torch.randn_like(changed[:,2])*100
        torch.testing.assert_close(a,candidate_summaries(changed,valid))
        torch.testing.assert_close(a,candidate_summaries(torch.cat([q,changed]),valid.expand(2,-1))[:1])
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(float(a[:,2].abs().max()),0)
        self.assertEqual(float(candidate_summaries(torch.zeros_like(q),valid).abs().max()),0)
        single=candidate_summaries(q[:,:1],valid[:,:1])
        self.assertTrue(torch.isfinite(single).all())
        torch.testing.assert_close(single[...,-2:],torch.zeros_like(single[...,-2:]),atol=1e-6,rtol=0)

    def test_training_hash_validation_standalone_reload_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=config_for(root)
            model=build_model(cfg)
            source=Path(cfg.TRAIN.FROZEN_FUSION.SOURCE_CHECKPOINT)
            torch.save(dict(model_name='RegionalSTAttention',model_arguments=model.extractor_arguments,
                            model_state=model.extractor.state_dict(),epoch=39,random_seed=100),source)
            cfg.defrost();cfg.TRAIN.FROZEN_FUSION.SOURCE_SHA256=hashlib.sha256(source.read_bytes()).hexdigest();cfg.freeze()
            wrong=cfg.clone();wrong.defrost();wrong.TRAIN.FROZEN_FUSION.SOURCE_SHA256='wrong';wrong.freeze()
            with self.assertRaisesRegex(ValueError,'SHA256'):
                initialize_frozen_source(model,wrong)
            bad=torch.load(source,weights_only=True);bad['model_arguments']['temporal_backend']='tcn'
            with self.assertRaisesRegex(ValueError,'architecture'):
                model.load_extractor_checkpoint(bad)
            _,_,rows=samples();loader=DataLoader(rows,batch_size=2)
            loaders={'train':loader,'valid':loader,'test':loader}
            with contextlib.redirect_stdout(io.StringIO()):
                trainer=RegionalDINOTrainer(cfg,loaders)
                before={k:v.clone() for k,v in trainer.core.extractor.state_dict().items()}
                trainer.train(loaders)
            for k,v in trainer.core.extractor.state_dict().items():
                torch.testing.assert_close(v,before[k],atol=0,rtol=0)
            self.assertEqual(len(list((root/'checkpoints').glob('*.pth'))),2)
            payload=torch.load(trainer.best_path,weights_only=True)
            self.assertEqual(payload['source_checkpoint']['sha256'],cfg.TRAIN.FROZEN_FUSION.SOURCE_SHA256)
            validate_model_checkpoint(payload,cfg)
            mismatch=cfg.clone();mismatch.defrost();mismatch.MODEL.REGIONAL_FROZEN_FUSION.GATE_MODE='moments';mismatch.freeze()
            with self.assertRaisesRegex(ValueError,'settings differ'):
                validate_model_checkpoint(payload,mismatch)
            source.unlink()
            test=cfg.clone();test.defrost();test.TOOLBOX_MODE='only_test';test.INFERENCE.MODEL_PATH=str(trainer.best_path);test.LOG.PATH=str(root/'test');test.freeze()
            with contextlib.redirect_stdout(io.StringIO()):
                reloaded=RegionalDINOTrainer(test,{'test':loader});reloaded.test({'test':loader})
            saved=torch.load(root/'test/regional_waveforms.pt',weights_only=True)
            self.assertEqual(saved['source_checkpoint']['epoch'],39)
            self.assertEqual(saved['model_name'],'RegionalFrozenFusion')

    def test_audit_standalone_new_checkpoint_preserves_source_lineage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=config_for(root);model=build_model(cfg)
            paths=[]
            for i in range(2):
                pulse=np.sin(2*np.pi*1.2*(np.arange(160)+160*i)/30)
                x=np.broadcast_to(120+pulse[:,None,None,None],(160,9,9,3)).astype('float32').copy()
                p=root/f'subject2_L1_MO1_E2_S4_GE1_GL1_H2_MA2_input{i}.npy'
                np.save(p,x);np.save(str(p).replace('_input','_label'),pulse.astype('float32'));paths.append(str(p))
            for split in ['train','valid']:
                with (root/f'{split}.csv').open('w',newline='') as f:
                    writer=csv.writer(f);writer.writerow(['input_files'])
                    writer.writerows([[p if split=='valid' else p.replace('subject2_','subject1_')] for p in paths])
            cfg.defrost();cfg.BASE=[''];cfg.TRAIN.DATA.FILE_LIST_PATH=str(root/'train.csv');cfg.VALID.DATA.FILE_LIST_PATH=str(root/'valid.csv');cfg.freeze()
            path=root/'audit.yaml';path.write_text(cfg.dump())
            checkpoint=root/'fusion.pth'
            torch.save(dict(model_name=cfg.MODEL.NAME,model_arguments=model_arguments(cfg),model_state=model.state_dict(),
                            epoch=7,source_checkpoint=dict(sha256='synthetic',epoch=39)),checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                result=audit_main(['--config',str(path),'--checkpoint',str(checkpoint),'--device','cpu',
                                   '--output',str(root/'audit'),'--plot-recordings','0','--save-waveforms'])
            self.assertEqual(result['evaluated_frames'],320)
            manifest=json.loads((root/'audit/manifest.json').read_text())
            self.assertEqual(manifest['source_checkpoint']['sha256'],'synthetic')


if __name__=='__main__':
    unittest.main()
