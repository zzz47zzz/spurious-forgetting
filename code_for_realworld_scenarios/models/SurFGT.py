import os
import numpy as np
import logging
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
from datasets import Dataset

from utils.metric import ResultSummary
from utils.backbone import get_backbone
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.buffer import get_buffer
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_sent_level_acc_with_generation
from models.Base import BaseLearner
# from utils.datatypes import STR2BOOL

logger = logging.getLogger()

def get_SurFGT_params(parser):
    '''
        The parameters of model SurFGT 
    '''
    parser.add_argument("--SurFGT_direction_bg_step", type=float, default=100, help="The begin step id for computing the forgetting direction")
    parser.add_argument("--SurFGT_direction_ed_step", type=float, default=150, help="The end step id for computing the forgetting direction")
    parser.add_argument("--SurFGT_average_n_times", type=int, default=10, help="The num of repetition for computing the forgetting direction")
    parser.add_argument("--SurFGT_component_list", type=list, 
                        default=['all'],
                        # ['mlp.dense_4h_to_h','mlp.dense_h_to_4h','attention.dense','attention.query_key_value','embed_in','embed_out'] 
                        help="The component for gradient projection")
    parser.add_argument("--SurFGT_freeze_component_list", type=list, 
                        default=[],
                        # ['mlp.dense_4h_to_h','mlp.dense_h_to_4h','attention.dense','attention.query_key_value','embed_in','embed_out'] 
                        help="The component for gradient projection")
    parser.add_argument("--SurFGT_freeze_bg_task_id", type=int, default=1, help="The task id begin freezing.")
    


class SurFGT(BaseLearner):
    '''
        Training a model sequentially to a number of tasks
    '''
    def __init__(self, params, CL_dataset, accelerator): 
        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['None'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'SurFGT')
        assert params.il_mode in ['CIT','IIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'SurFGT')

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        if self.params.il_mode == 'IIL':
            self.result_summary_train = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        
    def build_backbone(self):
        self.model, self.tokenizer = get_backbone(self.params)

    def build_classifier(self):
        self.classifier_list = []
        self.ce_loss = nn.CrossEntropyLoss()

    def build_optimizer(self):
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None
        if self.params.is_replay:
            self.buffer = get_buffer(self.params,self.CL_dataset.continual_config,self.accelerator)
    
    def accelerate_prepare(self):

        self.model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.model, self.optimizer, *self.train_loader_list)
        if len(self.dev_loader_list)>1:
            self.dev_loader_list = self.accelerator.prepare(*self.dev_loader_list)
            self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)
        else:
            self.dev_loader_list = [self.accelerator.prepare(*self.dev_loader_list)]
            self.test_loader_list = [self.accelerator.prepare(*self.test_loader_list)]
    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        # self.forgetting_direction = deepcopy(self.model)
        self.null_space_projection = {}
        super().begin_task(task_id)
        
    def end_task(self, task_id):
        super().end_task(task_id)

        if self.params.is_replay:
            self.buffer.update_buffer(task_id, self.train_loader_list[task_id], self.model, self.tokenizer)
    # ==============================================================================================

    # ================================= Epoch-Level Functions =======================================
    def train_epochs(self, task_id):
        '''
            Training the model with serveral epochs
        '''

        if task_id>0 and self.params.is_replay and not self.params.Replay_batch_level:
            train_dataset = self.train_loader_list[task_id].dataset
            buf_dataset = Dataset.from_dict(self.buffer.get_all_data())
            buf_dataset.set_format(type='torch',
                                   columns=train_dataset.format['columns'])
            cur_train_loader = DataLoader(
                ConcatDataset((train_dataset,buf_dataset)),
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False
            )
            cur_train_loader = self.accelerator.prepare(cur_train_loader)
        else:
            cur_train_loader = self.train_loader_list[task_id]

        # if task_id>0:
        #     # Record the Start Point
        #     init_model = deepcopy(self.model)

        #     # Record the similarity
        #     cosine_sim_dict = {}

        #     # Take of average of the direction from n "Preliminary Training"
        #     for rep_id in range(self.params.SurFGT_average_n_times):

        #         print(f'The {rep_id+1}-th preliminary training ...')
        #         # One Preliminary Training
        #         _step_id = 0

        #         cosine_sim_dict[rep_id] = {}

        #         while _step_id < self.params.SurFGT_direction_ed_step:

        #             for lm_input in cur_train_loader:

        #                 if _step_id == self.params.SurFGT_direction_bg_step:
        #                     _start_model = deepcopy(self.model)

        #                 # Sample from buffer and combine old data with new data
        #                 if task_id>0 and self.params.is_replay and self.params.Replay_batch_level:
        #                     buffer_lm_input = self.buffer.get_one_batch()
        #                     for k in lm_input.keys(): 
        #                         if k not in buffer_lm_input.keys():
        #                             continue
        #                         if isinstance(lm_input[k], list):
        #                             lm_input[k] =  lm_input[k] + buffer_lm_input[k]
        #                         elif isinstance(lm_input[k], torch.Tensor):
        #                             lm_input[k] = torch.cat((lm_input[k],buffer_lm_input[k].to(lm_input[k].device)),dim=0)
        #                         else:
        #                             raise NotImplementedError() 

        #                 # Compute loss
        #                 # Training with Causal Language Modeling Loss
        #                 total_loss = self.model(**{'input_ids':lm_input['input_ids_with_ans'], 
        #                                                 'attention_mask':lm_input['attention_mask_with_ans'],
        #                                                 'labels':lm_input['labels_with_ans']}).loss

        #                 # Backward
        #                 self.model.train()
        #                 self.optimizer.zero_grad()        
        #                 self.accelerator.backward(total_loss)
        #                 self.optimizer.step()

        #                 _step_id += 1

        #                 if _step_id >= self.params.SurFGT_direction_ed_step:
        #                     break

                
        #         # Compute Forgetting Direction
        #         for (n1,p1), (n2,p2), (n3,p3) in zip(self.forgetting_direction.named_parameters(), 
        #                                             _start_model.named_parameters(),
        #                                             self.model.named_parameters()):
        #             assert n1 == n2, n2==n3
        #             # TODO: Method 1, Method 4, Method 5: average direction
        #             if rep_id == 0:
        #                 p1.data = (p3.data - p2.data)/self.params.SurFGT_average_n_times
        #             else:
        #                 if len(p1.data.shape) == 2:
        #                     _cos_sim = F.cosine_similarity(p1.data*self.params.SurFGT_average_n_times/rep_id, 
        #                                         (p3.data - p2.data),
        #                                         dim=1).mean().item() # Row wise
        #                     cosine_sim_dict[rep_id][n1] = _cos_sim

        #                 p1.data += (p3.data - p2.data)/self.params.SurFGT_average_n_times

        #             # TODO: Method 3: Compute the Left NULL SPACE of 10 flatten weight matrix
        #             # if len(p1.data.shape) == 2:
        #             #     # Compute the left-null space
        #             #     if self.null_space_projection.get(n1,None) is None:
        #             #         self.null_space_projection[n1] = []
        #             #     self.null_space_projection[n1].append((p3.data - p2.data).flatten())


        #         _start_model.cpu()
        #         del _start_model

        #         self.model.cpu()
        #         del self.model
        #         del self.optimizer

        #         torch.cuda.empty_cache()
        #         self.accelerator.free_memory()

        #         # Undo the Preliminary Training
        #         self.model = deepcopy(init_model)
        #         self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)
        #         self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)         

        #     print(f'cosine_sim_dict = {cosine_sim_dict}')
        #     aver_cos_sim_list = []
        #     for rep_id in range(1,self.params.SurFGT_average_n_times):
        #         aver_cos_sim_list.append(np.mean([_cos for n, _cos in cosine_sim_dict[rep_id].items()]))
        #     print(f'aver_cos_sim_list = {aver_cos_sim_list}')

        #     init_model.cpu()
        #     del init_model

        #     torch.cuda.empty_cache()
        #     self.accelerator.free_memory()

        #     # TODO: Method 2: Compute the Left NULL SPACE 
        #     # for n, p in self.forgetting_direction.named_parameters():
        #     #     if len(p.data.shape) == 2 and not ('embed_in' in n or 'embed_out' in n) and max(p.data.shape)<10000:
        #     #         # Compute the left-null space
        #     #         U, S, VT = torch.svd(p.data)
        #     #         rank = self.get_empirical_rank(input_matrix=p.data, singular_matrix=S, thres=0.99)
        #     #         self.null_space_projection[n] = U[:,rank:] @ U[:,rank:].T

        #     # TODO: Method 3: Compute the Left NULL SPACE of 10 flatten weight matrix
        #     # for n, p in self.null_space_projection.items():
        #     #     stack_matrix = torch.stack(p, dim=0)
        #     #     U, S, VT = torch.svd(stack_matrix)
        #     #     rank = self.get_empirical_rank(input_matrix=stack_matrix, singular_matrix=S, thres=0.99)
        #     #     # self.null_space_projection[n] = VT[:,rank:] @ VT[:,rank:].T # NOTE: do not multiply in this stage, the dim is too large
        #     #     if rank == VT.shape[1]:
        #     #         rank = VT.shape[1]-1
        #     #     self.null_space_projection[n] = VT[:,rank:]

        # torch.cuda.empty_cache()
        # self.accelerator.free_memory()

        # Standard Training with Gradient Projection
        total_epochs = self.params.training_epochs

        for epoch_id in range(total_epochs):

            if self.accelerator.is_main_process:
                logger.info("------------------------ epoch %d ------------------------" %(epoch_id+1))

            self.begin_epoch(task_id, epoch_id)

            for lm_input in cur_train_loader:
                self.observe_batch(task_id, epoch_id, lm_input) 

            self.end_epoch(task_id, epoch_id)

    def begin_epoch(self, task_id, epoch_id):
        '''
            Start of each epoch
        '''
        # Avoid overwrite the result of the same global step
        if ((self.params.evaluate_interval>0) and (epoch_id>0 and epoch_id%self.params.evaluate_interval==0)) or \
            (self.params.is_evaluate_init and task_id==0 and epoch_id==0):
            self.evaluate_model(task_id=task_id)
        self.loss_list = []


    def observe_batch(self, task_id, epoch_id, lm_input):
        '''
            Observe a batch of data
        '''
        # Update step
        self.step += 1
        self.global_step += 1

        # Sample from buffer and combine old data with new data
        if task_id>0 and self.params.is_replay and self.params.Replay_batch_level:
            buffer_lm_input = self.buffer.get_one_batch()
            for k in lm_input.keys(): 
                if k not in buffer_lm_input.keys():
                    continue
                if isinstance(lm_input[k], list):
                    lm_input[k] =  lm_input[k] + buffer_lm_input[k]
                elif isinstance(lm_input[k], torch.Tensor):
                    lm_input[k] = torch.cat((lm_input[k],buffer_lm_input[k].to(lm_input[k].device)),dim=0)
                else:
                    raise NotImplementedError() 

        # Gradient accumulation
        with self.accelerator.accumulate(self.model):

            # Compute loss
            # Training with Causal Language Modeling Loss
            total_loss = self.model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                            'attention_mask':lm_input['attention_mask_with_ans'],
                                            'labels':lm_input['labels_with_ans']}).loss

            # Backward
            self.model.train()
            self.optimizer.zero_grad()        
            self.accelerator.backward(total_loss)

            # Gradient Projection
            # if task_id>0:
            #     for (n1, p1), (n2, p2) in zip(self.model.named_parameters(),self.forgetting_direction.named_parameters()):
            #         assert n1 == n2
            #         is_proj = False
            #         if 'all' in self.params.SurFGT_component_list:
            #             is_proj = True
            #         else:
            #             for _n in self.params.SurFGT_component_list:
            #                 if _n in n1:
            #                     is_proj = True
            #                     break
            #         if is_proj:
            #             # TODO: Method 1: Project Row Wise (Best)
            #             if len(p1.shape)==2:
            #                 p1.grad -= p2.data * F.cosine_similarity(p1.grad, p2.data, dim=1).unsqueeze(1)
            #             elif len(p1.shape)==1:
            #                 p1.grad -= p2.data * p1.grad.dot(p2.data)/p2.data.norm()
            #             # #  
            #             # TODO: Method 2: First SVD, then Project to null space (bad)
            #             # if len(p1.shape)==2 and self.null_space_projection.get(n1,None) is not None:
            #             #     p1.grad = self.null_space_projection[n1] @ p1.grad

            #             # TODO: Method 3: First SVD (flatten weight matrix), then Project to null space (bad)
            #             # if len(p1.shape)==2 and self.null_space_projection.get(n1,None) is not None:
            #             #     p1.grad = (self.null_space_projection[n1] @ (self.null_space_projection[n1].T @ p1.grad.flatten().unsqueeze(1))).flatten().reshape(p1.shape)
            
            #             # TODO: Method 4: Project Row Wise (Encourage Re-Alignment) (Worse, Not even Learning)
            #             # if len(p1.shape)==2:
            #             #     p1.grad = - p2.data * F.cosine_similarity(p1.grad, p2.data, dim=1).unsqueeze(1)
            #             # elif len(p1.shape)==1:
            #             #     p1.grad = - p2.data * p1.grad.dot(p2.data)/p2.data.norm()

            #             # TODO: Method 5: Project Row Wise (Encourage Undo Alignment)? (bad)
            #             # if len(p1.shape)==2:
            #             #     p1.grad = p2.data * F.cosine_similarity(p1.grad, p2.data, dim=1).unsqueeze(1)
            #             # elif len(p1.shape)==1:
            #             #     p1.grad = p2.data * p1.grad.dot(p2.data)/p2.data.norm()

            #         if len(self.params.SurFGT_freeze_component_list)>0:
            #             for _n in self.params.SurFGT_freeze_component_list:
            #                 if _n in n1:
            #                     p1.grad = p1.grad*0
            #                     break
            # TODO: Method 6: Just Freeze
            if task_id>=self.params.SurFGT_freeze_bg_task_id:
                for (n1, p1) in self.model.named_parameters():
                    for _n in self.params.SurFGT_freeze_component_list:
                        if _n in n1:
                            p1.grad = p1.grad*0
                            break

            scalar_loss = total_loss.item()
            if not(np.isnan(scalar_loss)) or not(np.isinf(scalar_loss)):
                self.optimizer.step()
                self.loss_list.append(scalar_loss)

        # Print training information
        if self.params.info_per_steps and self.step%self.params.info_per_steps==0:
            mean_loss = np.mean(self.loss_list)
            if self.accelerator.is_main_process:
                logger.info("Epoch %d, Step %d: Total_loss=%.3f,"%(
                        epoch_id+1, self.step, mean_loss
                ))
            self.accelerator.log({'loss':mean_loss},step=self.global_step)

        # save ckpt
        if self.params.save_llm_ckpt and self.params.save_llm_ckpt_step_interval!=-1 \
            and (self.global_step+1)%self.params.save_llm_ckpt_step_interval==0:
            self.model.save_pretrained(os.path.join(self.params.dump_path,'checkpoint_llm_task_%d_epoch_%d_step_%d'%(task_id,epoch_id,self.global_step)))
        

    def end_epoch(self, task_id, epoch_id):
        '''
            End of each epoch
        '''
        # Print training information
        if len(self.loss_list)>0:
            mean_loss = np.mean(self.loss_list)
            if self.accelerator.is_main_process:
                logger.info("Epoch %d, Step %d: Total_loss=%.3f"%(
                            epoch_id+1, self.step, mean_loss
                    ))
            self.accelerator.log({'loss':mean_loss},step=self.global_step)

        # For evaluation
        if (self.params.evaluate_interval>0) and epoch_id%self.params.evaluate_interval==0:
            il_mode = self.params.il_mode
            if il_mode == 'IIL':
                acc, save_dict = self.evaluate_current_task(task_id, task_id, 'dev', il_mode)
            else:
                acc = self.evaluate_current_task(task_id, task_id, 'dev', il_mode)
            if self.accelerator.is_main_process:
                logger.info("Mode %s, Current Task %d, Epoch %d, Step %d: Dev_acc=%.3f" % (
                    il_mode, task_id, epoch_id+1, self.step, acc
                ))
            self.accelerator.log({'Dev_Acc_Task_%d'%(task_id):acc},step=self.global_step)
            dev_score = acc

            if dev_score > self.best_score:
                if self.accelerator.is_main_process:
                    logger.info("Find better model!!")
                if self.params.save_llm_ckpt and self.params.save_llm_ckpt_best:
                    self.model.save_pretrained(os.path.join(self.params.dump_path,'checkpoint_llm_task_%d_epoch_best'%(task_id)))
        
        # save ckpt
        if self.params.save_llm_ckpt and self.params.save_llm_ckpt_epoch_interval!=-1 \
            and (epoch_id+1)%self.params.save_llm_ckpt_epoch_interval==0:
            self.model.save_pretrained(os.path.join(self.params.dump_path,'checkpoint_llm_task_%d_epoch_%d'%(task_id,epoch_id)))
        
        # Saving GPU memory
        torch.cuda.empty_cache()
    # ===========================================================================================


    # ================== Evaluation, Logging, Saving and Loading Functions ======================
    def evaluate_current_task(self,
                                eval_task_id: int, 
                                cur_task_id: int, 
                                phase: str,
                                il_mode: str,
                                return_dict: bool=False) -> dict:
        '''
            Evaluate the model on the current task

            Params: 
                - eval_task_id: the id of the task to be evaluated, 
                this information should NOT be provided to the CIL model during inference!
                - cur_task_id: the id recording how many tasks the model has learned,
                this information can be provided to the CIL model during inference.
                - phase: 'train','dev'or'test'
                - il_mode: 'CIT', 'IIL'

            Return:
                - acc:  accuracy (%) 
        '''

        assert phase in ['train','test','dev']
        assert il_mode in ['CIT','IIL'], 'NotImplemented for il_mode %s'%(il_mode)
        if phase=='train':
            data_loader = self.train_loader_list
        elif phase=='dev':
            data_loader = self.dev_loader_list
        else:
            data_loader = self.test_loader_list

        # NOTE: When not using classifier, the SEQ model does not need (benefit from) task identity 
        if self.params.backbone_max_new_token_list is not None:
            setattr(self.params,'backbone_max_new_token',self.params.backbone_max_new_token_list[eval_task_id])
            
        acc = evaluate_sent_level_acc_with_generation(
            model=self.model,
            eval_data_loader=data_loader[eval_task_id],
            tokenizer=self.tokenizer,
            accelerator=self.accelerator,
            params=self.params,
            idx2label=self.CL_dataset.continual_config.get('idx2label',None),
            metric=self.params.metric_for_each_task[eval_task_id] if self.params.metric_for_each_task is not None else 'acc' 
        )

        return  acc
  
    # ===========================================================================================


    # ============================================ Others ========================================
    def get_empirical_rank(self, input_matrix, singular_matrix, thres=0.99) -> int:
        '''
            Compute the empirical rank with threshold (e.g., 0.99)
        '''
        empirical_rank = 0
        accum_var = 0
        total_var = torch.norm(input_matrix, p="fro").pow(2).item()
        
        while accum_var < thres * total_var:
            accum_var += singular_matrix[empirical_rank].pow(2).item()
            empirical_rank += 1

        return empirical_rank
    
    # ===========================================================================================
