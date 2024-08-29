_base_ = [
    'mmseg::stdc/stdc1_4xb12-80k_cityscapes-512x1024.py',
    # '../../deploy_cfgs/mmcls/classification_openvino_dynamic-224x224.py'
]

# _base_.val_dataloader.batch_size = 32

test_cfg = dict(
    type='mmrazor.PTQLoop',
    calibrate_dataloader=_base_.val_dataloader,
    calibrate_steps=32,
)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

float_checkpoint = 'C:/Users/kosta/Documents/IPC/MMSegmentation/mmsegmentation/work_dirs/stdc1_4xb12-80k_cityscapes-512x1024/epoch_1.pth'  # path to my model

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor = dict(
        type='mmseg.SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    architecture=_base_.model,
    float_checkpoint=float_checkpoint,
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
        skipped_methods=[
                'mmseg.models.decode_heads.decode_head.BaseDecodeHead.predict_by_feat',
                'mmseg.models.decode_heads.decode_head.BaseDecodeHead.loss_by_feat']
            )
            ))

model_wrapper_cfg = dict(type='mmrazor.MMArchitectureQuantDDP', )
