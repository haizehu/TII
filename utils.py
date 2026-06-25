import sys

from tree_sitter import Language, Parser
from parsers import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parsers import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
max_functions = 30
max_packages = 30
reserved_info_slots = 5 # 保留信息位的数量

dfg_function = {
        'python': DFG_python,
        'java': DFG_java,
        'ruby': DFG_ruby,
        'go': DFG_go,
        'php': DFG_php,
        'javascript': DFG_javascript
    }
# 导入解析器
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parsers/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 nl_ids,
                 url,
                 package_ids,
                 function_ids,
                 levels

    ):
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg
        self.nl_ids = nl_ids
        self.url=url
        self.package_ids = package_ids
        self.function_ids = function_ids
        self.levels = levels

class InputFeaturesBase(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 url

    ):
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg
        self.url = url

def findIndex(levels):
    return levels[:max_functions] + [-1] * (max_functions - len(levels))

def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code,'utf8'))
        root_node = tree.root_node
        tokens_index=tree_to_token_index(root_node)
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)
        try:
            DFG,_=parser[1](root_node,index_to_code,{})
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg


def convert_examples_to_features(item):
    js, tokenizer, args = item

    # code
    parser = parsers[args.lang]
    # extract data flow
    code_tokens, dfg = extract_dataflow(js['original_string'], parser, args.lang)
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]


    temp_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(temp_tokens)

    # truncating
    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg = dfg[:args.code_length + args.data_flow_length - len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length

    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    # nl
    nl = ' '.join(js['docstring_tokens'])
    nl = ' Query: ' + nl
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]

    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)[:args.nl_length]
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    package_tokens = []
    package_ids = []
    for pack in js['package']:
        pack_token = tokenizer.tokenize(pack)[:args.context_length - 2]
        pack_token = [tokenizer.cls_token] + pack_token + [tokenizer.sep_token]
        pack_id = tokenizer.convert_tokens_to_ids(pack_token)
        padding_length = args.context_length - len(pack_id)
        pack_id += [tokenizer.pad_token_id] * padding_length
        package_tokens.append(pack_token)
        package_ids.append(pack_id)
    package_ids = package_ids[:max_packages]
    package_ids += [[tokenizer.pad_token_id] * args.context_length] * (max_packages - len(package_ids))

    # ==================== 寻找最优结构起点的最终逻辑 S T A R T ====================
    original_functions = js['function']
    original_levels = js['levels']
    up_fun_num = max(0, min(js['up_fun_num'] - 1, len(original_levels) - 1))  # 同时确保它不是负数

    if len(original_levels) > max_functions:
        # 1. 定义候选区域：这个区域是长度为30、且以 up_fun_num 为结尾的。
        search_start = max(0, up_fun_num - max_functions + 1)
        search_end = up_fun_num + 1

        # 2. 在候选区域内，寻找 level 最小的索引
        best_start_index = up_fun_num
        min_level_found = original_levels[up_fun_num] if up_fun_num < len(original_levels) else sys.maxsize

        # 从后往前遍历候选区域，寻找更好的（level更小）起点
        for i in range(search_end - 2, search_start - 1, -1):
            if original_levels[i] <= min_level_found:
                min_level_found = original_levels[i]
                best_start_index = i

        # 3. 确定最终的窗口
        start_index = best_start_index
        end_index = start_index + max_functions

        # Python切片会自动处理结尾超出边界的情况
        processed_functions = original_functions[start_index:end_index]
        processed_levels = original_levels[start_index:end_index]
    else:
        # 如果未超出，则直接使用原始列表
        processed_functions = original_functions
        processed_levels = original_levels
    # ==================== 寻找最优结构起点的最终逻辑 E N D ======================

    # 在处理过的列表上进行后续操作
    levels = findIndex(processed_levels)
    # 2. 额外加入5个-1作为信息位
    levels += [-1] * reserved_info_slots

    function_tokens = []
    function_ids = []
    if processed_functions:
        for func in processed_functions:
            func_token = tokenizer.tokenize(func)[:args.context_length - 2]
            func_token = [tokenizer.cls_token] + func_token + [tokenizer.sep_token]
            func_id = tokenizer.convert_tokens_to_ids(func_token)
            padding_length = args.context_length - len(func_id)
            func_id += [tokenizer.pad_token_id] * padding_length
            function_tokens.append(func_token)
            function_ids.append(func_id)

    # 填充 function_ids 至固定长度
    function_ids += [[tokenizer.pad_token_id] * args.context_length] * (max_functions - len(function_ids))
    # 2. 额外加入5个虚拟函数作为固定填充
    function_ids += [[tokenizer.pad_token_id] * args.context_length] * reserved_info_slots

    return InputFeatures(code_ids, position_idx, dfg_to_code, dfg_to_dfg, nl_ids, js['url'], package_ids, function_ids,
                         levels)

def convert_examples_to_features_base(item):
    js,tokenizer,args=item
    #code
    parser=parsers[args.lang]
    #extract data flow
    code_tokens,dfg=extract_dataflow(js['original_string'],parser,args.lang)
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))
    code_tokens=[y for x in code_tokens for y in x]
    #truncating
    code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=args.code_length+args.data_flow_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
    return InputFeaturesBase(code_ids,position_idx,dfg_to_code,dfg_to_dfg, js['url'])

