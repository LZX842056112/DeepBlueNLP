import copy
import multiprocessing as mp


def process_configs(target, arg_parser):
    args, _ = arg_parser.parse_known_args()
    ctx = mp.get_context('spawn')

    for run_args, _run_config, _run_repeat in _yield_configs(arg_parser, args):
        p = ctx.Process(target=target, args=(run_args,))  # 创建一个子进程
        p.start()  # 启动子进程
        p.join()  # 等待子进程执行完成


def _read_config(path):
    lines = open(path, "r", encoding="utf-8").readlines()

    runs = []
    run = [1, dict()]
    for line in lines:
        stripped_line = line.strip()

        # continue in case of comment
        if stripped_line.startswith('#'):
            continue

        # 取出#前的内容
        sid = stripped_line.find('#')
        if sid > 0:
            stripped_line = stripped_line[:sid].strip()

        if not stripped_line: # 判断是否遇到空行，如果遇到空行表示新的参数组合
            if run[1]:
                runs.append(run)

            run = [1, dict()]
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs


def _convert_config(config):
    config_list = []
    for k, v in config.items():
        if v.lower() == 'true':
            config_list.append('--' + k)
        elif v.lower() != 'false':
            config_list.extend(['--' + k] + v.split(' '))

    return config_list


def _yield_configs(arg_parser, args, verbose=True):
    _print = (lambda x: print(x)) if verbose else lambda x: x

    if args.config:
        config = _read_config(args.config)

        for run_repeat, run_config in config:
            print("-" * 50)
            print("Config:")
            print(run_config)

            args_copy = copy.deepcopy(args)
            config_list = _convert_config(run_config)
            run_args = arg_parser.parse_args(config_list, namespace=args_copy)
            run_args_dict = vars(run_args)

            # set boolean values
            for k, v in run_config.items():
                if v.lower() == 'false':
                    run_args_dict[k] = False

            print("Repeat %s times" % run_repeat)
            print("-" * 50)

            for iteration in range(run_repeat):
                _print("Iteration %s" % iteration)
                _print("-" * 50)

                yield run_args, run_config, run_repeat

    else:
        yield args, None, None
