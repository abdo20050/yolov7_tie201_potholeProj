import torch
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import torch_pruning as tp
from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import torch.nn.utils.prune as prune
from models.yolo import Detect, IDetect 

global device



def load_model(weights_path, dev, prune_amount=0):
    global device 
    device = dev
    # model = torch.load(weights_path, map_location=device)['model'].float()
    weights = weights_path
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # model.to(device).eval()
    model.train(True)
    for param in model.parameters():
        param.requires_grad = True

    ################################################################################
    # Pruning
    if prune_amount > 0:
    #     example_inputs = torch.randn(1, 3, 224, 224).to(device)
    #     imp = tp.importance.MagnitudeImportance(p=2) # L2 norm pruning

    #     ignored_layers = []
    #     from models.yolo import IDetect
    #     for m in model.modules():
    #         # print(m)
    #         if isinstance(m, IDetect):
    #             ignored_layers.append(m)
    #     print(ignored_layers)

    #     iterative_steps = 1 # progressive pruning
    #     pruner = tp.pruner.MagnitudePruner(
    #         model,
    #         example_inputs,
    #         importance=imp,
    #         iterative_steps=iterative_steps,
    #         pruning_ratio=prune_amount, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    #         ignored_layers=ignored_layers,
    #     )
    #     base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        

    #     # pruner.step()
    #     # pruner.summary()
    #     pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    #     # # print(model)
    #     print("Before Pruning: MACs=%f G, #Params=%f G"%(base_macs/1e9, base_nparams/1e9))
    #     print("After Pruning: MACs=%f G, #Params=%f G"%(pruned_macs/1e9, pruned_nparams/1e9))

        ignored_layers = []
        for m in model.modules():
            if isinstance(m, (Detect, IDetect)):
                ignored_layers.append(m)

        # Function to apply structured pruning to Conv2d layers
        def apply_structured_pruning(model, amount=0.5, n=1, dim=0):
            for name, module in model.named_modules():
                if module in ignored_layers:
                    pass
                    # print(module)
                    # prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
                    # # Optionally, remove the pruning reparameterization to make pruning permanent
                    # prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Conv2d):
                    # print(module)
                    prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
                # prune 40% of connections in all linear layers
                elif isinstance(module, torch.nn.Linear):
                    # print(module)
                    prune.l1_unstructured(module, name='weight', amount=amount)
        # Apply pruning
        apply_structured_pruning(model, amount=prune_amount, n=1, dim=0)
    ####################################################################################

    model = TracedModel(model, device, 640)
    # model.half() 

    return model


def preprocess_frame(frame, img_size=640):
    img = letterbox(frame, img_size, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device)
    # img_tensor = img_tensor.half()
    img_tensor = img_tensor.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def run_inference(model, img_tensor, frame, device):

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, 0.1, 0.45, classes=None, agnostic=False)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{int(cls)} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)
    return frame
