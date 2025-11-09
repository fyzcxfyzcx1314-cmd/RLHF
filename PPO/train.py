from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 创建Dataset
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.final_prompts = []

        for prompt in prompts:
            if apply_chat_template:
                content = [{"role" : "user", "content" : prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize = False, add_generation_prompt = True)
            else:
                prompt = self.tokenizer.bos_token + prompt
            self.final_prompts.append(prompt)
    def __len__(self):
        return len(self.final_prompts)
    def __getitem__(self, index):
        return self.final_prompts[index]

# critic model
class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
    def forward(self, inputs_id, attention_mask, num_actions):
        hidden_size = self.base_model(inputs_id, attention_mask = attention_mask).last_hidden_state
        value_head_output = self.value_head(hidden_size)
        value = value_head_output.squeeze(-1)[:, : -num_actions]
        return value

# Actor loss
def compute_policy_loss(log_probs, old_log_probs, advantages, attention_mak=None, clip_eps=0.2):
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2)
    if attention_mak:
        return ((loss * attention_mak).sum(-1)) / (attention_mak.sum(-1)).mean()
    else:
        return loss.mean(-1).mean()

# Critic loss(returns 是实际收益)
def compute_value_loss(values, old_values, returns, attention_mask=None, clip_eps: float=None):
    if clip_eps:
        values_clipped = old_values + (values - old_values).clamp(1.0 - clip_eps, 1.0 + clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2
    
    if attention_mask:
        return ((loss * attention_mask).sum(-1)) / (attention_mask.sum(-1)).mean()
    else:
        return (loss * attention_mask).sum(-1).mean()

# 经验池（存储模型生成数据，供之后使用）
class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = {
            "seqs",
            "action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
            "num_actions"
        }

        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
        self.buffer.extend(batch)
        if len(self.buffer) > self.limit:
            self.buffer = self.buffer[len(self.buffer) - self.limit:]
    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)
    def clear(self):
        self.buffer = []
    def __getitem__(self, index):
        return self.buffer[index]

# 样本容器，统一格式
@dataclass
class Samples:
    seqs : torch.Tensor
    attention_mask : Optional[torch.LongTensor]
    action_mask : Optional[torch.BoolTensor]
    num_actions : Union[int, torch.Tensor]
    packed_seq_lens : Optional[torch.Tensor]
    response_length : torch.Tensor
    total_length : torch.Tensor

@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None

@dataclass
class BufferItem:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]

# 优势：A(t) = R(t) + gam*V(t+1) - V(t)
# 考虑当前优势和未来优势：A(t) = R(t) + gam*V(t+1) - V(t) + gam*lam*A(t+1)
# 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
# A(T-1) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T) 知道A(T)可计算A(T-1) 依次类推
# 优化的实际收益：returns(t) = A(t) + V(t) = = R(t) + gam * (V(t+1) + lam * A(t+1))
def get_advantages_and_returns(values, rewards, action_mask, gamma, lambd):
    # 下个时间步的优势A(t + 1)
    lastgaelem = 0
    # 方向计算的结果
    advantages_reversed = []
    # 时间步数
    response_length = rewards.zize(1)

    if action_mask is not None:
        values = values * action_mask
        rewards = rewards * action_mask
    
    for t in reversed(range(response_length)):
        # A(t + 1)
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        # 优势
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        # GAE, 考虑当前和未来优势，并利用最后一个未来优势逆推
        lastgaelem = delta + gamma * lambd * lastgaelem
        advantages_reversed.append(lastgaelem)
    advantages = torch.stack(advantages_reversed[::-1], dim = -1)
    returns = advantages + values
    return returns, advantages.detach()

def compute_approx_kl(log_probs: torch.Tensor, 
                      ref_log_probs: torch.Tensor,
                      action_mask: Optional[torch.Tensor] = None):
    log_ratio = log_probs - ref_log_probs
    if action_mask:
        log_ratio = log_ratio * action_mask
    return log_ratio

def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
    kl_divergence_estimate = -kl_ctl * kl
    rewards = kl_divergence_estimate

    ends = action_mask.sum(1) + 1
    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(device)
    reward_clip = torch.slamp(r, -clip_reward_value, clip_reward_value)
    batch_size = r.size(0)
    for j in range(batch_size):
        rewards[j, :ends[j]][-1] += reward_clip[j, 0]
    return rewards

# 生成样本
def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    sample_list = []
    model.eval()
    all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in prompts], [])
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        prompts = all_prompts[i : i + micro_rollout_batch_size]
        inputs = actor_tokenizer(prompts, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        seqs = model.generate(**inputs.to(device),
                             max_new_tokens = max,
                             pad_token_id = pad_token_id,
                             eos_token_id = eos_token_id
                             )
        if seqs.size(1) >= max_length + max_new_tokens:
            seqs = seqs[:, : max_length + max_new_tokens]
        else:
            seqs = torch.cat([seqs, torch.full((seqs.size(0), max_new_tokens + max_length - seqs.size(1)), fill_value=pad_token_id, device=seqs.device)], dim = 1)
        
        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        ans = seqs[:, input_ids.size(1)]
        action_mask = (ans.ne(eos_token_id) & ans.ne(pad_token_id)).to(dtype=torch.long)

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
        sample_list.append(samples)
    return sample_list

# 生成经验
def generate_experiences(samples_list):
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []

    for sample in samples_list:
        seqs = sample.seqs
        attention_mask = sample.attention_mask
        action_mask = sample.action_mask
        num_actions = sample.num_actions

        with torch.no_grad():
            # 计算action model输出的token概率
            output = actor_model(seqs, attenion_mask = attention_mask)
            logits = output.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim = -1)
            log_probs_labels = log_probs.gather(dim = -1, index = seqs[:, 1:].unsqueeze(-1))
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
            # 计算reference model输出的概率
            ref_output = ref_model(seqs, attenion_mask = attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim = -1)
            ref_log_probs_labels = ref_log_probs.gather(dim = -1, index = seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]
            # 计算value
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)
            # 计算奖励
            seq_text = actor_tokenizer.batch_decode(seqs, skip_special_tokens = True)
            reward_model_inputs = reward_tokenizer(seq_text, return_tensor = 'pt', padding = True)
            r = reward_model(**reward_model_inputs.to(device)).logits
            # 计算kl散度
            kl = compute_approx_kl(action_log_probs, ref_action_log_probs, action_mask = action_mask).to(device)
            # 计算实际奖励
            rewards = compute_rewards(kl, r, action_mask, kl_ctl = 0.1, clip_reward_value = 0.2)
            # 计算优势和回报
            rewards, advantages = get_advantages_and_returns(value, rewards, action_mask, gamma = 0.1, lambd = 0.2)
        #actor_model.train()
        #critic_model.train()

        experience = Experience(
            action_log_probs.detach(),
            value.detach(),
            advantages.detach(),
            attention_mask,
            action_mask,
            r.detach(),
            sample.responce_length,
            sample.total_length,
            num_actions,
            kl.detach()
        )
        experiences.append(experience)
    return experiences

def collate_fn(batch):
    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []

    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])
    
    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)

    return BufferItem(seqs, action_log_probs, values, advantages, attention_mask, action_mask, action_mask.size(1))


def train_step(experiences, steps):
    actor_model.train()
    actor_optimizer.zero_grad()

    sequences = experiences.seqs
    old_action_log_probs = experiences.action_log_probs
    advantages = experiences.advantages
    num_actions = experiences.num_actions
    attention_mask = experiences.attention_mask
    action_mask = experiences.action_mask
    old_values = experiences.values
    returns = experiences.returns

    logits = actor_model(sequences, attention_mask = attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim = -1)
    log_probs_labels = log_probs.gather(dim = -1, index = sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

    # 计算actor loss
    policy_loss = compute_policy_loss(
        action_log_probs,
        old_action_log_probs,
        advantages,
        action_mask = attention_mask
    )
    policy_loss.backward()
    actor_optimizer.step()

    writer.add_scalar("policy_loss", policy_loss.item(), steps)

    # 更新critic model
    critic_model.train()
    critic_optimizer.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    values_loss = compute_value_loss(values, old_values, returns, action_mask)
    values_loss.backward()
    critic_optimizer.step()
    writer.add_scalar("values_loss", values_loss, steps)

    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {values_loss.item():.4f}")

# 训练流程
def train():
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            samples = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size)
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(
                buffer,
                batch_size = micro_train_batch_size,
                collate_fn = collate_fn,
                shuffle=True
            )
            torch.cuda.empty_cache()
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
            buffer.clear()

            torch.cude.empty_cache()

if __name__ == "main":
    device = "cuda" if torch.cude.is_available() else "cpu"
    # 迭代轮次
    episodes = 3
    # 一次经验计算轮次
    max_epochs = 5
    # batch_size
    rollout_batch_size = 8
    # 样本的batch_size
    micro_rollout_batch_size = 2
    # 一个提示词生成多少样本
    n_samples_per_prompt = 2
    # 最大生成token数
    max_new_tokens = 50
    # 生成的最大长度
    max_length = 256
    # 实际训练的batch_size
    micro_train_batch_size = 2
    # 记录日志
    writer = SummaryWriter('./runs')
    # actor模型
    actor_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    # reference模型
    ref_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    # 奖励模型
    reward_model = AutoModelForSequenceClassification.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2').to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    # critic mox
    critic_model = Critic(actor_model.base_model).to(device)
    # 优化器
    actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr = 0.00005)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr = 0.00005)
    # 填充
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id
    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    #数据集
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(
        prompts_dataset,
        batch_size = rollout_batch_size,
        shuffle=True
    )

    train()