import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.data import HeteroData, Batch

class GatingFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatingFusion, self).__init__()
        self.hidden_size = hidden_size

        # 定义用于计算门控信号的 MLP
        # 输入是两个向量的拼接 (2 * hidden_size)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),  # 使用平滑的激活函数增加非线性
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()  # Sigmoid 函数确保输出值在 (0, 1) 区间，作为门控信号
        )

        # 添加一个输出MLP，对融合后的向量做进一步的非线性变换
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, nl_vec, context_embs):
        # 1. 将两个向量拼接，作为计算门控信号的输入
        combined_vec = torch.cat([nl_vec, context_embs], dim=1)

        # 2. 通过 MLP 计算门控信号 gate
        gate = self.gate_mlp(combined_vec)

        # 3. 应用门控机制进行加权融合
        #    gate 越接近 1，nl_vec 的权重越大
        #    gate 越接近 0，context_embs 的权重越大
        gated_fused_vec = gate * nl_vec + (1 - gate) * context_embs

        #    将融合后的信息作为对原始 nl_vec 的一个增量更新
        output_vec = self.output_mlp(gated_fused_vec)
        final_vec = self.layer_norm(gated_fused_vec + output_vec)

        return final_vec

class ContextGNNEncoderHetero(nn.Module):
    def __init__(self, hidden_size=768, gat_heads=4, gat_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.gat_layers = gat_layers
        self.gat_heads = gat_heads

        self.in_linear_dict = nn.ModuleDict({
            'query': nn.Linear(hidden_size, hidden_size),
            'package': nn.Linear(hidden_size, hidden_size),
            'function': nn.Linear(hidden_size, hidden_size),
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(gat_layers):
            conv = HeteroConv({
                # 1. 局部调用关系 (只在真实function间)
                ('function', 'calls', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                             add_self_loops=False),
                ('function', 'is_called_by', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                    add_self_loops=False),

                # 2. 全局枢纽关系 (真实function <--> level=-1 function)
                ('function', 'gathers_from', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                    add_self_loops=False),
                ('function', 'broadcasts_to', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                     add_self_loops=False),

                # 3. 其他关系保持不变
                ('package', 'in_context_of', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                    add_self_loops=False),
                ('query', 'attends_to', 'package'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                              add_self_loops=False),
                ('query', 'attends_to', 'function'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                               add_self_loops=False),
                ('function', 'context_for', 'package'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                  add_self_loops=False),
                ('package', 'attended_by', 'query'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                               add_self_loops=False),
                ('function', 'attended_by', 'query'): GATv2Conv(-1, hidden_size // gat_heads, heads=gat_heads,
                                                                add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)

            # 为每种节点类型都创建一个 LayerNorm 实例
            norm_dict = nn.ModuleDict({
                'query': nn.LayerNorm(hidden_size),
                'package': nn.LayerNorm(hidden_size),
                'function': nn.LayerNorm(hidden_size),
            })
            self.norms.append(norm_dict)
        self.output_mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, batched_data: Batch):
        """
        直接处理一个批量化的图对象 (Batch object)。
        """
        x_dict = batched_data.x_dict
        edge_index_dict = batched_data.edge_index_dict

        edge_index_dict_to_use = edge_index_dict

        # --- 2. GNN 聚合 ---
        # a) 应用初始线性变换和激活函数
        for node_type, x in x_dict.items():
            # x_dict[node_type] = self.in_linear_dict[node_type](x).relu()
            x_dict[node_type] = F.gelu(self.in_linear_dict[node_type](x))

        # b) 执行多层GNN卷积
        for i, conv in enumerate(self.convs):
            x_dict_input = x_dict

            # 边进行卷积
            x_dict_updates = conv(x_dict_input, edge_index_dict_to_use)

            # 应用完整的更新流程: 残差 -> 归一化 -> 激活
            for node_type, x_update in x_dict_updates.items():
                # 1. 残差连接
                x_res = x_dict_input[node_type] + x_update
                # 2. 归一化 (LayerNorm)
                x_norm = self.norms[i][node_type](x_res)
                # 3. 激活 (ReLU)
                # x_activated = x_norm.relu()
                x_activated = F.gelu(x_norm)
                # 4. 正则化 (Dropout)
                x_dict[node_type] = x_activated

        # --- 3. 获取最终输出 ---
        query_final_emb = x_dict['query']
        context_vec = self.output_mlp(query_final_emb)

        return context_vec


class DynamicLayerSelector(nn.Module):
    """动态层选择模块"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        attn_dim = hidden_size // 2

        self.layer_attention = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.ReLU(),  # 更强非线性
            nn.Linear(attn_dim, attn_dim // 2),
            nn.GELU(),  # 再加一层非线性
            nn.Linear(attn_dim // 2, 1)
        )

    def attention_based_selection(self, all_hidden_states):
        """基于注意力机制的层选择"""
        # all_hidden_states: [B, L, H]
        B, L, H = all_hidden_states.size()

        # 计算每层注意力权重
        layer_scores = self.layer_attention(all_hidden_states).squeeze(-1)  # [B, L]
        layer_weights = F.softmax(layer_scores, dim=1)  # [B, L]

        weighted_hidden = torch.bmm(layer_weights.unsqueeze(1), all_hidden_states).squeeze(1)

        return weighted_hidden

class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.embeddings = encoder.embeddings

        # 获取隐藏层大小
        hidden_size = 768

        # 初始化两个独立的 DynamicLayerSelector
        self.nl_layer_selector = DynamicLayerSelector(
            hidden_size=hidden_size
        )
        self.code_layer_selector = DynamicLayerSelector(
            hidden_size=hidden_size
        )

        self.context_gnn_encoder = ContextGNNEncoderHetero(hidden_size=768, gat_heads=4, gat_layers=2)

        self.FusionModule = GatingFusion(hidden_size=hidden_size)

    def encode_layerwise(self, output, encoder_type=None):
        all_hidden_states = output.hidden_states[1:]  # 去掉 embedding 层
        all_cls = torch.stack([layer[:, 0, :] for layer in all_hidden_states], dim=1)  # [B, L, H]

        if self.training:
            B, L, H = all_cls.size()
            # 每个样本生成一个独立的打乱顺序
            perm = torch.stack([torch.randperm(L) for _ in range(B)], dim=0).to(all_cls.device)  # [B, L]
            batch_idx = torch.arange(B).unsqueeze(1).expand(-1, L).to(all_cls.device)  # [B, L]
            all_cls = all_cls[batch_idx, perm, :]  # [B, L, H]

        if encoder_type == 'nl_dynamic_selection':
            weighted_hidden = self.nl_layer_selector.attention_based_selection(all_cls)
            return weighted_hidden

        elif encoder_type == 'code_dynamic_selection':
            weighted_hidden = self.code_layer_selector.attention_based_selection(all_cls)
            return weighted_hidden

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None, context_inputs=None, levels=None):
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)

            # 从token ids计算embeddings（正常训练）
            inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1, keepdim=True) + 1e-10)
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

            output = self.encoder(inputs_embeds=inputs_embeddings,
                                  attention_mask=attn_mask,
                                  position_ids=position_idx,
                                  output_hidden_states=True)
            code_vec = self.encode_layerwise(output, encoder_type='code_dynamic_selection')
            # code_vec = self.code_projection_head(code_vec)
            return code_vec

        elif nl_inputs is not None:
            # 从token ids计算（正常训练）
            output = self.encoder(nl_inputs,
                                    attention_mask=nl_inputs.ne(1),
                                    output_hidden_states=True)
            nl_vec = self.encode_layerwise(output, encoder_type='nl_dynamic_selection')
            if levels is None:
                return nl_vec
            # 上下文处理
            return self._process_context_batched(nl_vec, context_inputs, levels)

    def _process_context_batched(self, nl_vec, context_inputs, levels):
        """
        使用 PyTorch Geometric 的 Batch 机制进行批量化处理。
        """
        package_inputs, function_inputs = context_inputs
        batch_size = package_inputs.shape[0]

        with torch.no_grad():
            # 步骤 1: 压平输入，为批量处理做准备
            all_package_inputs = package_inputs.view(-1, package_inputs.size(-1))
            all_function_inputs = function_inputs.view(-1, function_inputs.size(-1))

            # --- 处理 package 上下文 ---
            package_encodings_flat = self.encoder(all_package_inputs,
                                                  attention_mask=all_package_inputs.ne(1))[1]

            # --- 处理 function 上下文 ---
            function_encodings_flat = self.encoder(all_function_inputs,
                                                   attention_mask=all_function_inputs.ne(1))[1]

            # ======================= 修改结束 ============================

            # 步骤 3: 恢复原始的批次结构
            package_encodings = package_encodings_flat.view(batch_size, -1, package_encodings_flat.size(-1))
            function_encodings = function_encodings_flat.view(batch_size, -1, function_encodings_flat.size(-1))

        data_list = []
        for i in range(batch_size):
            # 提取单个样本的数据
            query_emb = nl_vec[i]
            # pkg_emb = package_encodings[i]
            func_emb = function_encodings[i]
            lvl = levels[i]

            # 2. 过滤占位的 package 节点
            valid_pkg_mask = package_inputs[i][:, 0] != 1
            # 使用掩码过滤 package embedding
            pkg_emb = package_encodings[i][valid_pkg_mask]

            # 为当前样本创建一个 HeteroData 对象
            data = HeteroData()

            # a) 添加节点特征
            data['query'].x = query_emb.unsqueeze(0)
            data['package'].x = pkg_emb
            data['function'].x = func_emb

            num_packages = pkg_emb.size(0)
            num_functions = func_emb.size(0)

            # b) 添加边索引 (这里的逻辑与你原来的GNN forward方法中构建边的逻辑完全相同)
            # 关系 1: package <--> function
            if num_packages > 0 and num_functions > 0:
                p_indices = torch.arange(num_packages)
                f_indices = torch.arange(num_functions)
                grid_p, grid_f = torch.meshgrid(p_indices, f_indices, indexing='ij')
                edge_index_p_f = torch.stack([grid_p.flatten(), grid_f.flatten()], dim=0)

                # 正向边: package -> in_context_of -> function
                data['package', 'in_context_of', 'function'].edge_index = edge_index_p_f
                # 反向边: function -> context_for -> package (通过交换行列实现)
                data['function', 'context_for', 'package'].edge_index = edge_index_p_f[[1, 0]]

            # 关系 2: function <--> function，拆分为局部调用和全局关系
            if num_functions > 1:
                levels_tensor = torch.as_tensor(lvl, device=nl_vec.device)

                # 1. 找出真实节点和占位节点的全局索引
                real_func_indices = (levels_tensor != -1).nonzero(as_tuple=True)[0]
                pad_func_indices = (levels_tensor == -1).nonzero(as_tuple=True)[0]

                # 2. 【局部调用边】只在真实函数之间创建 'calls' / 'is_called_by' 边
                if len(real_func_indices) > 1:
                    real_levels = levels_tensor[real_func_indices]
                    adj_matrix_calls = real_levels.unsqueeze(1) < real_levels.unsqueeze(0)
                    # 获取在 real_func_indices 内部的相对索引
                    src_local, dst_local = adj_matrix_calls.nonzero(as_tuple=True)
                    # 将相对索引映射回在 func_emb 中的全局索引
                    src_global = real_func_indices[src_local]
                    dst_global = real_func_indices[dst_local]
                    edge_index_calls = torch.stack([src_global, dst_global], dim=0)

                    data['function', 'calls', 'function'].edge_index = edge_index_calls
                    data['function', 'is_called_by', 'function'].edge_index = edge_index_calls[[1, 0]]

                # 3. 【全局枢纽边】在真实函数与占位函数之间创建专用边
                if len(real_func_indices) > 0 and len(pad_func_indices) > 0:
                    # 使用meshgrid快速创建所有真实节点到所有占位节点的连接
                    grid_real, grid_pad = torch.meshgrid(real_func_indices, pad_func_indices, indexing='ij')
                    edge_index_real_to_pad = torch.stack([grid_real.flatten(), grid_pad.flatten()], dim=0)

                    # 定义专用边：信息汇聚 (真实节点 -> 全局节点)
                    data['function', 'gathers_from', 'function'].edge_index = edge_index_real_to_pad
                    # 定义专用边：信息广播 (全局节点 -> 真实节点)
                    data['function', 'broadcasts_to', 'function'].edge_index = edge_index_real_to_pad[[1, 0]]

            # 关系 3: query <--> package
            if num_packages > 0:
                q_indices = torch.zeros(num_packages, dtype=torch.long)
                p_indices = torch.arange(num_packages)
                edge_index_q_p = torch.stack([q_indices, p_indices], dim=0)

                # 正向边: query -> attends_to -> package
                data['query', 'attends_to', 'package'].edge_index = edge_index_q_p
                # 反向边: package -> attended_by -> query
                data['package', 'attended_by', 'query'].edge_index = edge_index_q_p[[1, 0]]

            # 关系 4: query <--> function
            if num_functions > 0:
                q_indices = torch.zeros(num_functions, dtype=torch.long)
                f_indices = torch.arange(num_functions)
                edge_index_q_f = torch.stack([q_indices, f_indices], dim=0)

                # 正向边: query -> attends_to -> function
                data['query', 'attends_to', 'function'].edge_index = edge_index_q_f
                # 反向边: function -> attended_by -> query
                data['function', 'attended_by', 'query'].edge_index = edge_index_q_f[[1, 0]]

            data_list.append(data)

        # 2. 将图对象列表打包成一个大的批量图
        batched_data = Batch.from_data_list(data_list).to(nl_vec.device)

        # 3. 单次调用GNN编码器
        context_embs = self.context_gnn_encoder(batched_data)  # GNN的forward需要修改

        # 融合
        fused_vec = self.FusionModule(nl_vec, context_embs)
        # return F.normalize(fused_vec, p=2, dim=-1)
        return fused_vec
