import sys
import time

import math

from config import set_args
from utils import convert_examples_to_features,convert_examples_to_features_base

import os
import torch
import random
import json
import pickle
import numpy as np
import multiprocessing
from tqdm import tqdm
from model_context_GNN_OPT_2_batch import Model
from datetime import datetime
from torch.optim import AdamW
from torch import nn
from torch import amp
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup, RobertaConfig, RobertaTokenizer, RobertaModel
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel
from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler

import subprocess

# 设置空闲显存阈值（单位：MB）
MEMORY_FREE_THRESHOLD_MB = 1024  # 例如至少空出 13GB 才启动任务

def get_gpu0_free_memory():
    """返回 GPU 0 的空闲显存（MB）"""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return int(output.strip().split('\n')[0])
    except Exception as e:
        print("无法获取 GPU 显存:", e)
        return 0

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        # 取语言名（倒数第二个目录）
        lang = file_path.split('/')[-2]
        # 取文件名前缀
        prefix = file_path.split('/')[-1][:-5]
        # 拼接缓存文件名
        cache_file = os.path.join(args.output_dir, f"{lang}_{prefix}.pkl")
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for line in tqdm(f):
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data), desc="Processing examples"))
            pickle.dump(self.examples, open(cache_file, 'wb'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].nl_ids),
                torch.tensor(self.examples[item].package_ids),
                torch.tensor(self.examples[item].function_ids),
                torch.tensor(self.examples[item].levels))

class TextDatasetBase(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        # 取语言名（倒数第二个目录）
        lang = file_path.split('/')[-2]
        # 取文件名前缀
        prefix = file_path.split('/')[-1][:-5]
        # 拼接缓存文件名
        cache_file = os.path.join(args.output_dir, f"{lang}_{prefix}.pkl")
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for line in tqdm(f):
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
            self.examples = pool.map(convert_examples_to_features_base, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx))


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def write_arg_log(s, output_dir):
    logs_path = os.path.join(output_dir, 'arg_log.txt')
    with open(logs_path, 'a+', encoding='utf-8') as f:
        f.write(f"\n==== Log at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        for key in sorted(s.keys()):
            f.write(f"{key} : {s[key]}\n")
    print(f"日志文件 {logs_path} 已创建成功。")

def hfedr_loss(nl_vecs, code_vecs):
    """
    nl_vecs: [B, H] 自然语言向量
    code_vecs: [B, H] 代码向量
    """
    B, H = nl_vecs.shape

    # 2. 相似度矩阵
    scores = (nl_vecs @ code_vecs.T) # [B, B]
    labels = torch.arange(B, device=nl_vecs.device)
    loss = F.cross_entropy(scores, labels)
    return loss


def load_checkpoint(model, optimizer, scheduler, load_path, device=None):
    checkpoint = torch.load(load_path, map_location=device ,weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    print(f"Loaded checkpoint from {load_path}, resume from epoch {start_epoch}")
    return model, optimizer, scheduler, start_epoch


def train_model(args, model, tokenizer, pool):
    # 构建数据集和加载器
    train_dataset = TextDataset(tokenizer, args, args.train_data, pool=pool)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        pin_memory=True
    )

    # 计算训练总步数
    total_steps = len(train_dataloader) * args.num_train_epochs
    # 设置 warmup 步数（一般占总步数5%-10%）
    warmup_steps = int(0.05 * total_steps)

    # 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=total_steps,
    #     num_cycles=0.5  # 半个cosine周期，默认即可
    # )
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # 多卡训练支持
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(args.device)

    # 初始化保存路径

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    best_model_dir = os.path.join(args.output_dir, "best_mrr_model")
    os.makedirs(best_model_dir, exist_ok=True)

    bestmodel = "best_mrr_model/model_epoch_.pt"
    checkpoint_path = os.path.join(args.output_dir, bestmodel)

    if os.path.exists(checkpoint_path):
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device=args.device
        )
        print(f"从第 {start_epoch} 轮开始继续训练")
    else:
        print("没有找到 checkpoint，从头开始训练")
        start_epoch=0

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Total train batch size = {args.train_batch_size}")
    print(f"  Total optimization steps = {len(train_dataloader) * args.num_train_epochs}")

    # AMP混合精度初始化
    scaler = amp.GradScaler("cuda")
    optimizer.zero_grad()
    model.train()

    tr_loss = 0.0
    tr_num = 0

    for epoch in range(start_epoch, args.num_train_epochs):

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            code_ids, attn_mask, position_idx, nl_ids, package_ids, function_ids, levels = batch
            code_ids = code_ids.to(args.device)
            attn_mask = attn_mask.to(args.device)
            position_idx = position_idx.to(args.device)
            nl_ids = nl_ids.to(args.device)
            package_inputs = package_ids.to(args.device)
            function_inputs = function_ids.to(args.device)
            # levels, routes 等无需传 device，model 内部处理即可

            with amp.autocast("cuda"):
                code_vecs = model(code_inputs=code_ids, attn_mask=attn_mask, position_idx=position_idx)
                nl_vecs = model(nl_inputs=nl_ids, context_inputs=[package_inputs, function_inputs], levels=levels)
                total_loss = hfedr_loss(nl_vecs, code_vecs)

                tr_loss += total_loss.item()
                tr_num += 1
                if (step + 1) % 100 == 0 or step==0:
                    tqdm.write(f"Epoch {epoch+1} step {step+1} loss {tr_loss / tr_num:.5f}")
                    tr_loss = 0
                    tr_num = 0

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # 每轮保存模型
        epoch_model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }, epoch_model_path)
        print(f"Epoch {epoch+1} model saved to {epoch_model_path}")

        # 删除上一轮保存的模型
        if epoch > 0:
            previous_model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            if os.path.exists(previous_model_path):
                os.remove(previous_model_path)
                print(f"Removed previous model: {previous_model_path}")

        evaluating(args, model, tokenizer, epoch, pool)

def evaluating(args, model, tokenizer, epoch, pool):
    print("开始评估模型")
    model_path = f"outputs_model/model_epoch_{epoch+1}.pt"
    print(f"\n评估模型: {model_path}")
    # 只加载模型参数即可
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    print(f"Loaded model from {model_path}")

    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(args, model, tokenizer, args.valid_data , pool)
        print(f"Evaluation Results for epoch {epoch+1}:")
        print(f"MRR   : {metrics['eval_mrr']:.4f}")
        print(f"Top1  : {metrics['top1']:.4f}")
        print(f"Top5  : {metrics['top5']:.4f}")
        print(f"Top10 : {metrics['top10']:.4f}")
        print(f"Top100: {metrics['top100']:.4f}")
        metrics = evaluate_model(args, model, tokenizer, args.test_data, pool)
        print(f"Evaluation Results for Test_data")
        print(f"MRR   : {metrics['eval_mrr']:.4f}")
        print(f"Top1  : {metrics['top1']:.4f}")
        print(f"Top5  : {metrics['top5']:.4f}")
        print(f"Top10 : {metrics['top10']:.4f}")
        print(f"Top100: {metrics['top100']:.4f}")

def test_modle(args, model, tokenizer, pool):
    print("开始评估最好模型")
    model_path = f"outputs_model/model_epoch_104.pt"
    print(f"\n评估模型: {model_path}")
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    print(f"Loaded model from {model_path}")

    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(args, model, tokenizer, args.test_data, pool)
        print(f"Evaluation Results for best epoch:")
        print(f"MRR   : {metrics['eval_mrr']:.4f}")
        print(f"Top1  : {metrics['top1']:.4f}")
        print(f"Top5  : {metrics['top5']:.4f}")
        print(f"Top10 : {metrics['top10']:.4f}")
        print(f"Top100: {metrics['top100']:.4f}")

def evaluate_model(args, model, tokenizer, data_path, pool):
    model.eval()
    # 构造查询集（自然语言 + 上下文）
    query_dataset = TextDataset(tokenizer, args, data_path, pool=pool)
    query_dataloader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.eval_batch_size,
        pin_memory=True
    )

    # 构造代码集
    code_dataset = TextDatasetBase(tokenizer, args, args.codebase_file, pool=pool)
    code_dataloader = DataLoader(
        code_dataset,
        sampler=SequentialSampler(code_dataset),
        batch_size=args.eval_batch_size,
        pin_memory=True
    )

    # 提取 query 向量
    nl_vecs = []
    for batch in tqdm(query_dataloader, desc="Encoding queries"):
        code_ids, attn_mask, position_idx, nl_ids, package_ids, function_ids, levels = batch
        nl_inputs = nl_ids.to(args.device)
        package_inputs = package_ids.to(args.device)
        function_inputs = function_ids.to(args.device)
        with torch.no_grad(), amp.autocast("cuda"):
            nl_vec = model(nl_inputs = nl_inputs, context_inputs=[package_inputs, function_inputs], levels=levels)
            # nl_vec, _ = model(nl_inputs=nl_ids)
            # nl_vec = F.normalize(nl_vec, dim=-1)
            # nl_vec, _ = model(nl_inputs=nl_ids)
            # nl_vec = nl_vec.mean(dim=1)
            # nl_vecs.append(nl_vec.cpu().numpy())
            nl_vecs.append(nl_vec.detach())
    # nl_vecs = np.concatenate(nl_vecs, axis=0)
    nl_vecs = torch.cat(nl_vecs, dim=0)

    # 提取 code 向量
    code_vecs = []
    for batch in tqdm(code_dataloader, desc="Encoding codebase"):
        code_ids, attn_mask, position_idx = batch
        code_ids = code_ids.to(args.device)
        attn_mask = attn_mask.to(args.device)
        position_idx = position_idx.to(args.device)
        with torch.no_grad(), amp.autocast("cuda"):
            code_vec = model(code_inputs=code_ids, attn_mask=attn_mask, position_idx=position_idx)
            # code_vec = F.normalize(code_vec, dim=-1)
            # code_vec, _ = model(code_inputs=code_ids, attn_mask=attn_mask, position_idx=position_idx)
            # code_vec = code_vec.mean(dim=1)
            # code_vecs.append(code_vec.cpu().numpy())
            code_vecs.append(code_vec.detach())
    # code_vecs = np.concatenate(code_vecs, axis=0)
    code_vecs = torch.cat(code_vecs, dim=0)

    model.train()

    # 提取 URL 用于匹配
    nl_urls = [example.url for example in query_dataset.examples]
    code_urls = [example.url for example in code_dataset.examples]

    # MRR & Top-k 计算
    ranks = []
    top1, top5, top10, top100 = 0, 0, 0, 0

    # 定义一个处理查询的批次大小，可以根据你的显存/内存调整
    batch_size = args.eval_batch_size

    for i in tqdm(range(0, len(nl_vecs), batch_size), desc="Calculating scores"):
        # 取出一个批次的 query 向量
        nl_batch_vecs = nl_vecs[i:i + batch_size]

        # 计算这个批次的 query 与整个 codebase 的分数
        # 使用 torch.matmul 在 GPU 上计算，速度更快
        scores = torch.matmul(nl_batch_vecs.half(), code_vecs.T.half())

        # 对分数进行排序，并移回 CPU 处理
        sort_ids = torch.argsort(scores, dim=-1, descending=True).cpu().numpy()

        # 获取这个批次对应的 URL
        batch_urls = nl_urls[i:i + batch_size]

        for url, sort_id in zip(batch_urls, sort_ids):
            rank = 0
            find = False
            # 只检查前 1000 个结果
            for idx in sort_id[:1000]:
                if not find:
                    rank += 1
                if code_urls[idx] == url:
                    find = True
                    break  # 找到后立即退出内层循环

            if find:
                ranks.append(1 / rank)
                if rank <= 1: top1 += 1
                if rank <= 5: top5 += 1
                if rank <= 10: top10 += 1
                if rank <= 100: top100 += 1
            else:
                ranks.append(0)

    results = {
        "eval_mrr": float(np.mean(ranks)),
        "top1": float(top1 / len(ranks)),
        "top5": float(top5 / len(ranks)),
        "top10": float(top10 / len(ranks)),
        "top100": float(top100 / len(ranks))
    }
    return results

def main(pool):
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 读参
    args = set_args()

    # 设置随机种子
    set_seed(args.seed)

    # 写训练日志
    write_arg_log(vars(args), args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # 设定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    gpu_num = torch.cuda.device_count()
    print(f"device: {device}, gpu_num: {gpu_num}")

    # 模型准备
    config = RobertaConfig.from_pretrained(args.pretrained_model_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model_path)
    model = RobertaModel.from_pretrained(args.pretrained_model_path)
    model=Model(model)

    # 开始训练
    print("开始训练")
    # 将模型加载到指定的设备上
    model.to(device)

    # 调用train_model函数进行训练
    train_model(args, model, tokenizer, pool)
    # test_modle(args, model, tokenizer, pool)

if __name__ == "__main__":
    print("等待 GPU 空闲（空闲显存超过 {} MB）...".format(MEMORY_FREE_THRESHOLD_MB))
    while True:
        free_mem = get_gpu0_free_memory()
        msg = f"\r当前 GPU 0 空闲显存: {free_mem} MB"
        sys.stdout.write(msg)
        sys.stdout.flush()

        if free_mem >= MEMORY_FREE_THRESHOLD_MB:
            print("\n显卡空闲，开始执行主程序。")
            break

        time.sleep(60)
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    with multiprocessing.Pool(cpu_count) as pool:
        main(pool)