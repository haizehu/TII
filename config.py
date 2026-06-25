import argparse
import os

def set_args():
    parser = argparse.ArgumentParser('--HFGCS')

    # 设置语言参数
    parser.add_argument('--lang', default='python', type=str, help='language')
    # 先解析 lang 参数
    args, _ = parser.parse_known_args()

    # 根据 lang 构建路径
    lang = args.lang
    train_path = os.path.join('./data', lang, 'train.json')
    valid_path = os.path.join('./data', lang, 'valid.json')
    test_path = os.path.join('./data', lang, 'test.json')
    codebase_path = os.path.join('./data', lang, 'codebase.json')

    parser.add_argument('--train_data', default=train_path, type=str, help='train_data path')
    parser.add_argument('--valid_data', default=valid_path, type=str, help='valid_data path')
    parser.add_argument('--test_data', default=test_path, type=str, help='test_data path')
    parser.add_argument("--codebase_file", default=codebase_path, type=str, help="codebase path")

    parser.add_argument('--pretrained_model_path', default='./graphcodebert-base', type=str, help='pretrained_model_path') #./roberta_pretrain
    parser.add_argument('--output_dir', default='./outputs_model', type=str, help='output_dir')

    parser.add_argument('--num_train_epochs', default=40 , type=int, help='num_train_epochs')

    parser.add_argument('--train_batch_size', default=256, type=int, help='train_batch_size')
    parser.add_argument('--eval_batch_size', default=256, type=int, help='eval_batch_size')

    parser.add_argument('--learning_rate', default=2e-5, type=float, help='learning_rate')  # 原2e-5
    parser.add_argument('--seed', default=123456, type=int, help='seed')

    parser.add_argument("--nl_length", default=128, type=int, help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int, help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--context_length", default=20, type=int, help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int, help="Optional Data Flow input sequence length after tokenization.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay')
    return parser.parse_args()