"""Read defaults from the pinned official entry, including shell overrides.

Only literal expressions are interpreted. No official training/import side effects
are executed when resolving a configuration.
"""
import ast
import configparser
import json
import operator
import re
import shlex


def value(node, names):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [value(n, names) for n in node.elts]
    if isinstance(node, ast.Dict):
        return {value(k, names):value(v, names) for k,v in zip(node.keys,node.values)}
    if isinstance(node, ast.Attribute):
        return getattr(value(node.value,names),node.attr)
    if isinstance(node, ast.Subscript):
        return value(node.value,names)[value(node.slice,names)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -value(node.operand,names)
    if isinstance(node, ast.BinOp):
        operation = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
                     ast.Div:operator.truediv, ast.Pow:operator.pow}[type(node.op)]
        return operation(value(node.left,names),value(node.right,names))
    raise ValueError(f'Unsupported default expression: {ast.dump(node)}')


def argparse_defaults(path):
    result = {}
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='add_argument':
            continue
        if not node.args or not isinstance(node.args[0],ast.Constant):
            continue
        options = {k.arg:k.value for k in node.keywords}
        name = node.args[0].value.lstrip('-').replace('-','_')
        if 'default' in options:
            result[name] = value(options['default'],{})
        elif 'action' in options and value(options['action'],{})=='store_true':
            result[name] = False
    return result


def shell_defaults(path, base):
    """Read literal NAME=value and --argument $NAME without running bash."""
    text = '\n'.join(line for line in path.read_text().splitlines() if not line.lstrip().startswith('#'))
    variables = dict(re.findall(r'^([A-Z][A-Z0-9_]*)=(.*)$',text,re.MULTILINE))
    tokens = shlex.split(text.replace('\\\n',' '))
    result = dict(base)
    for i,token in enumerate(tokens[:-1]):
        if token.startswith('--') and not tokens[i+1].startswith('--'):
            raw = tokens[i+1]
            if raw.startswith('$'):
                raw = variables[raw[1:].strip('{}')].strip('"\'')
            try:
                raw = ast.literal_eval(raw)
            except (ValueError,SyntaxError):
                pass
            result[token[2:].replace('-','_')] = raw
    return result


def json_comments(path):
    # The pinned INGP image config contains line comments and no URL strings.
    return json.loads('\n'.join(line.split('//',1)[0] for line in path.read_text().splitlines()))


def official(method, repo):
    if method=='siren':
        defaults = argparse_defaults(repo/'experiment_scripts/train_img.py')
        cls = next(n for n in ast.parse((repo/'modules.py').read_text()).body
                   if isinstance(n,ast.ClassDef) and n.name=='SingleBVPNet')
        init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        settings = {a.arg:value(d,{}) for a,d in zip(init.args.args[-len(init.args.defaults):],init.args.defaults)}
        settings['hidden_layers'] = settings.pop('num_hidden_layers')
        settings.pop('type')
        settings.pop('mode')
        return dict(entry='experiment_scripts/train_img.py -> training.train',model=settings,
                    training=dict(lr=defaults['lr'],epochs=defaults['num_epochs'],optimizer='Adam'))
    if method=='finer':
        defaults = shell_defaults(repo/'run_finer.sh',argparse_defaults(repo/'train_image.py'))
        return dict(entry='run_finer.sh -> train_image.py',model=dict(
            hidden_layers=defaults['hidden_layers'],first_omega_0=defaults['first_omega'],
            hidden_omega_0=defaults['hidden_omega'],first_bias_scale=defaults['first_bias_scale'],
            scale_req_grad=defaults['scale_req_grad']),
            training=dict(lr=defaults['lr'],epochs=defaults['num_epochs']))
    if method=='miner':
        cfg = configparser.ConfigParser()
        cfg.read(repo/'configs/image_32x32.ini')
        settings = {}
        for section in cfg.sections():
            settings[section] = {}
            for k,v in cfg[section].items():
                try:
                    v = ast.literal_eval(v)
                except (ValueError,SyntaxError):
                    pass
                settings[section][k] = v
        return dict(entry='configs/image_32x32.ini',config=settings)
    if method=='neural_experts':
        import yaml
        return dict(entry='configs/config_RGB.yaml',config=yaml.safe_load((repo/'configs/config_RGB.yaml').read_text()))
    if method=='ingp':
        return dict(entry='scripts/run.py -> configs/image/base.json',config=json_comments(repo/'configs/image/base.json'))
    raise ValueError(method)
