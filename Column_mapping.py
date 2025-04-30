import pandas as pd

# Define mapping rules
prefix_map = {
    'packet_length': 'payload_bytes',
    'iat': 'packets_IAT'
}

direction_map = {
    'biflow': '',  # total
    'upstream_flow': 'fwd',  # forward
    'downstream_flow': 'bwd'  # backward
}

stat_map = {
    'min': 'min',
    'max': 'max',
    'mean': 'mean',
    'std': 'std',
    'var': 'variance'
}


def translate_column(col_name):
    # Handle label columns
    if col_name in ('BF_label', 'BF_activity'):
        return col_name  # keep as-is

    # Extract parts
    parts = col_name.split('_')

    # Special handling for packet_length which has 4 parts
    if parts[0] == 'packet' and parts[1] == 'length':
        prefix = 'packet_length'
        direction = '_'.join(parts[2:-1])
        stat = parts[-1]
    else:  # iat has 3 parts
        prefix = parts[0]
        direction = '_'.join(parts[1:-1])
        stat = parts[-1]

    # Build target name
    prefix_target = prefix_map.get(prefix, '')
    direction_target = direction_map.get(direction, '')
    stat_target = stat_map.get(stat, '')

    if prefix_target and stat_target:
        if direction_target:
            return f'{direction_target}_{prefix_target}_{stat_target}'
        else:
            return f'{prefix_target}_{stat_target}'
    else:
        return None  # unmatched



