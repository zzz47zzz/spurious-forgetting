import torch
import torch.nn as nn

def get_optimizer(params, model, classifier_list):

    # build optimizer
    tg_params = []
    if isinstance(model,nn.Module) and params.method == 'SEQ':
        # If params.SEQ_fix_encoder and params.SEQ_warmup_epoch_before_fix_encoder>0, the encoder will be fixed by setting its learning rate as zero
        if not (params.SEQ_fix_encoder and params.SEQ_warmup_epoch_before_fix_encoder==0):
            if len(params.SEQ_freeze_component_list)!=0:
                param_list = []
                freeze_param_list = []
                for n,p in model.named_parameters():
                    is_freeze = False
                    for _freeze_n in params.SEQ_freeze_component_list:
                        if _freeze_n in n:
                            is_freeze = True
                            break
                    if not is_freeze:
                        param_list.append(p)
                    else:
                        freeze_param_list.append(p)
                # NOTE: The first parameter group is the params for not freezing         
                tg_params.append({'params': param_list, 'lr': float(params.lr), 'weight_decay': float(params.weight_decay)}) 
                # NOTE: The second parameter group is the params for not freezing         
                tg_params.append({'params': freeze_param_list, 'lr': float(params.lr), 'weight_decay': float(params.weight_decay)}) 
            else:
                tg_params.append({'params': model.parameters(), 'lr': float(params.lr), 'weight_decay': float(params.weight_decay)}) 

    elif isinstance(model,nn.Module):
        tg_params.append({'params': model.parameters(), 'lr': float(params.lr), 'weight_decay': float(params.weight_decay)}) 
    
    else:
        raise NotImplementedError()

    if isinstance(classifier_list,nn.ModuleList):
        for i in range(len(classifier_list)):
            tg_params.append({'params': classifier_list[i].parameters(), 'lr': float(params.classifier_lr)})

    optimizer = torch.optim.AdamW(tg_params)

    return optimizer