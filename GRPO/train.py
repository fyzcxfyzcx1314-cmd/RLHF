from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 加载数据
class GSM8KDataset(Dataset):
    def __init__(self, tokenizer, data_path):
        data = data_path.load_dataset(data_path)
        self.tokenizer = tokenizer
        self.data = data['train']
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = self.data[index]
        prompt = sample['question_zh-cn']
        answer = sample['answer_only']
        return {'prompt':prompt, 'answer':answer}

@dataclass
class Sample:
    prompt_response_ids : torch.Tensor
    response_ids : torch.Tensor
    prompts : Any
    answer : Any
    # prompt + response的掩码
    attention_mask : Optional[torch.LongTensor]
    # response的掩码
    action_mask : Optional[torch.BoolTensor]
    num_actions : Union[int, torch.Tensor]
    response_length : int

class GRPOArguments:
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4 # 组内样本数
    max_prompt_length = 256 # 最大输入长度
    max_generate_length = 256 # 最大输出长度
    reward_weights : List[float] = None # 奖励的权重（多个奖励函数）
    beta = 0.0 # KL散度的系数，为0则忽略KL散度，即不使用参考模型
    clip_eps = 0.2
    gradient_accumulation_steps = 2 # 梯度累加
    num_iterations = 1 # 采样一次样本训练模型轮数
    batch_size = 1

class GRPOtrainer:
    def __init__(self, model, reward_funcs, args, train_dataset, 
                 eval_dataset, tokenizer, reward_tokenizers):
        self.args = args
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)

        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
        else:
            self.ref_model = None

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = self.get_tokenizer(tokenizer)

        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(reward_func, num_labels = 1).to(self.args.device)
        self.reward_funcs = reward_funcs

        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to the number of reward_funcs.")
        
        self.reward_tokenizers = reward_tokenizers
        self.optimizer = torch.optim.Adam(self.model.parameter(), self.args.lr)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # 缓存一个batch的数据，供模型多次训练
        self.buffer = [None] * self.args.gradient_accmulation_steps
        self.updata_steps = 0
    
    def get_tokenizer(self, tokenizer):
        tokenizer.padding_size = "left"
        return tokenizer
    
    def generate_sample(self, inputs):
        sample_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(inputs)
        if 'ansewr' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        max_length = self.args.max_prompt_length + self.args.max_generate_length
        for prompt, answer in zip(prompts, answers):
            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, {"role": "user", 'content': prompt}], add_generation_prompt=True, tokenize=False)

            inputs = self.tokenizer([input_text] * self.args.num_generations, padding = 'max_length', max_length = self.args.max_prompt_length, return_tensor = 'pt')
            prompt_ids = inputs['input_ids']
            with torch.no_grad():
                prompt_response_ids = self.model(**inputs.to(self.args.device),
                                                 max_new_tokens = self.args.max_generate_length,
                                                 temperature = 0.9,
                                                 top_p = 1,
                                                 top_k = 50)
            if prompt_response_ids.size(0) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full((prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)], dim=1)
            
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_tokem_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (response_ids.ne(self.tokenizer.pad_token_id) & response_ids.ne(self.tokenizer.eos_token_id)).to(dtype=torch.long)

            sample = Sample(
                prompt_response_ids = prompt_response_ids,
                response_ids = response_ids,
                prompts = prompt,
                answer = answers,
                attention_mask = attention_mask,
                action_mask = action_mask,
                num_actions = action_mask.size(1),
                response_length = action_mask.float().sum(dim = -1)
            )
            sample_list.append(sample)
        return sample_list
    
    def generate_experiences(self, inputs):
        self.model.eval()
        sample_list = self.generate_sample(inputs)

        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        batch_advantages = []

        for samples in sample_list:
            prompt_response_ids = samples.prompt_response_ids
            response_ids = samples.response_ids
            prompt = samples.prompts
            answer = samples.answer
            attention_mask = samples.attention_mask
            action_mask = samples.action_mask
            num_actions = samples.num_actions
            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)

            with torch.no_grad():
                # 计算旧的模型生成的每个token的概率
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)

                if self.ref_model:
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)

                # 存储各个奖励函数在一个group内各个响应的奖励
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device = self.args.device)

                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens = True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]

                for i, (reward_func, reward_tokenizer) in enumerate(zip(self.reward_funcs, self.reward_tokenizers)):
                    if isinstance(reward_func, PreTrainedModel):
                        reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensor = 'pt', padding = True)
                        rewards_per_func[i]  = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    else:
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(prompt = prompt_texts, response = response_texts, answer = answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype = torch.float32, device = self.args.device)

                 # 奖励权重的计算
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype = torch.float32, device = self.args.device).unsqueeze(1)

                rewards = rewards.sum(dim = 0)
                mean_group = rewards.mean()
                std_group = rewards.std()

                # 句子粒度的GRPO
                advantage = (rewards - mean_group) / (std_group + 1e-8)
                batch_advantages.append(advantage)
        
        return {
            "prompt_response_ids" : torch.cat(batch_prompt_response_ids, dim = 0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }

    def compute_loss(self, model, inputs):
        
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        if self.args.beta != 0.0:
            
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            
            # k3: log_ratio.exp() - 1 - log_ratio
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']
        
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        coef_1 = torch.exp(action_log_probs - old_action_log_probs) # 重要性采样 shape: [batch_size * num_generations, num_actions]
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # 一个序列中每个token的优势是一样的
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1) # shape: [batch_size * num_generations]
        loss = loss.mean()
        
        # loss = per_token_loss.sum() / action_mask.sum()
        
        return loss

    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        output = model(input_ids, attention_mask = attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim = -1)
        log_probs_labels = log_probs.gather(dim = -1, index = input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs
    
    def train_step(self, model, inputs, optimizer, step):
        model.train()
        # scaler = torch.amp.GradScaler()
        # with torch.amp.autocast(device_type='cuda'):
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        # loss = scaler.scale(loss)
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            
            optimizer.step()
            optimizer.zero_grad()
            # scaler.unscale_(optimizer)
            # scaler.step(optimizer)
            # scaler.update()
        
            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        torch.cuda.empty_cache()

    def train(self):
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for _ in range(self.args.epoch):
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):
                
                inputs = self.generate_experiences(batch)
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                   
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                        
                del inputs
    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir) 

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    SYSTEM_PROMPT = """
        按照如下格式回答问题：
            <think>
                你的思考过程
            </think>
            <answer>
                你的回答
            </answer>
                    """
    
    args = GRPOArguments()
    
    writer = SummaryWriter('./runs')
    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    # 奖励函数
    # reward_model = '/home/user/Downloads/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    

    
    
    prompts_dataset = GSM8KDataset('/home/user/wyf/deepseek_learn/gsm8k_chinese', tokenizer)
  
    trainer = GRPOtrainer(model=model,
                          reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()