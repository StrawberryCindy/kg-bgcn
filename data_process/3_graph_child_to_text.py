import os
import pandas as pd
import pickle as pk
import json
import shutil


node_parameters = {
    ':DEVICE': ['id', 'labels', 'NAME', 'CORPID', 'AREA',  'IP', 'MAC',
                'WORKTYPE', 'CHILDIP', 'parent_devicecode', 'PRODUCT_TYPE',
                'NMAPSYSTEM', 'BELONGEDMAJOR', 'SYSTEMVERSION'
                ],
    ':DEVICECHILD': ['id', 'labels', 'IP', 'CORPID', 'safe_area', 'MAC',
                     'SERVICENAME', 'OPENPORTS',
                     'SCANPORTS', 'PORTS', 'NETWORK_CARD_VENDOR', 'PORT',
                     'SERVICEPRODUCT',
                     ],
    ':MESSAGE': ['id', 'labels', 'date', 'destIp', 'destPort', 'srcIp',
                 'srcPort', 'count', 'length', 'applicationProto',
                 'networkProto'
                 ],
}


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_child_name(device_name, file_list):
    if '/' in device_name:
        device_name = device_name.replace('/', '-')
    if '\n' in device_name:
        device_name = device_name.replace('\n', '')
    for file_name in file_list:
        if device_name in file_name:
            return file_name
    return None


def find_child_id(device_id, file_list):
    for file_name in file_list:
        file_id = file_name.split('.')[0]
        if str(device_id) == file_id:
            return file_name
    return None


def _get_device(node_dict):
    for node_id, node_info in node_dict.items():
        if str(node_info['labels']) in [':DEVICE']:
            return node_info
    return None


def _get_ip(node_dict):
    ip_list = []
    for node_id, node_info in node_dict.items():
        if str(node_info['labels']) in [':IP']:
            ip_list.append(node_info)
    return ip_list


def get_device_info(device_node):
    data_list = []
    keys = node_parameters[':DEVICE']
    did = device_node[keys[0]]
    label = device_node[keys[1]]
    result = f'{label}节点，'
    for key in keys[2:]:
        if key in device_node.keys():
            value = device_node[key]
            if value:
                result += f'{key}为{value}，'
    result = result[:-1] + '。'
    # print(len(result))
    info = {
        'id': did,
        'text': result,
        'type': label
    }
    data_list.append(info)
    return data_list


def get_short_info(device_node, node):
    data_list = []
    device_name = device_node['name']
    did = node['id']
    label = str(node['labels'])
    result = f'设备{device_name}的{label}节点，'
    for key in node.keys():
        if key in ['id', 'labels']:
            continue
        value = node[key]
        if value:
            result += f'{key}为{value}，'
    result = result[:-1] + '。'
    info = {
        'id': did,
        'text': result,
        'type': label
    }
    data_list.append(info)
    return data_list


def get_devicechild_info(device_node, node):
    data_list = []
    device_name = device_node['name']
    keys = node_parameters[':DEVICECHILD']
    did = node[keys[0]]
    label = node[keys[1]]
    result = f'设备{device_name}的{label}节点，'
    for key in keys[2:]:
        if key in node.keys():
            value = node[key]
            if value:
                result += f'{key}为{value}，'
    result = result[:-1] + '。'
    info = {
        'id': did,
        'text': result,
        'type': label
    }
    data_list.append(info)
    return data_list


def get_port_info(device_node, node, ip_list):
    data_list = []
    port_name = node['name']
    flag = False
    for ip in ip_list:
        if ip['name'] in port_name:
            flag = True
            break
    if not flag:
        return data_list
    data_list = get_short_info(device_node, node)
    return data_list


def get_other_port_info(device_node, node):
    data_list = []
    device_name = device_node['name']
    did = node['id']
    label = str(node['labels'])
    result = f'设备{device_name}和其他设备的{label}节点进行通信，'
    for key in node.keys():
        if key in ['id', 'labels']:
            continue
        value = node[key]
        if value:
            result += f'{key}为{value}，'
    result = result[:-1] + '。'
    info = {
        'id': did,
        'text': result,
        'type': label
    }
    data_list.append(info)
    return data_list


def get_device_name_info(device_node, node):
    data_list = []
    device_name = device_node['name']
    node_name = node['name']
    did = node['id']
    label = str(node['labels'])
    result = f'设备{device_name}和{node_name}进行通信。'
    info = {
        'id': did,
        'text': result,
        'type': label
    }
    data_list.append(info)
    return data_list


def get_message_info(device_node, node, relation_dict, port_dict):
    data_list = []
    find_list = []
    device_name = device_node['name']
    keys = node_parameters[':MESSAGE']
    node_id = node['id']
    label = node[keys[1]]
    for _, relation in relation_dict.items():
        start = relation['start']
        end = relation['end']
        if start == node_id:
            type_m = '发送'
            if port_dict.get(end):
                find_list.append(type_m)
        if end == node_id:
            type_m = '接收'
            if port_dict.get(start):
                find_list.append(type_m)
    if find_list:
        if len(find_list) == 1:
            result = f'设备{device_name}{find_list[0]}{label}节点，'
        else:
            result = f'设备{device_name}具有{label}节点，'
        for key in keys[2:]:
            if key in node.keys():
                value = node[key]
                if value:
                    result += f'{key}为{value}，'
        result = result[:-1] + '。'
        info = {
            'id': node_id,
            'text': result,
            'type': label
        }
        data_list.append(info)
    return data_list


def parse_node_in_graph(graph_data):
    node_dict = graph_data['node_dict']
    relation_dict = graph_data['relation_dict']
    key_list = graph_data['key_list']
    info_list = []
    port_dict = {}
    device_node = _get_device(node_dict)
    info_list = get_device_info(device_node)
    ip_list = _get_ip(node_dict)
    for node_id, node_info in node_dict.items():
        lables = str(node_info['labels'])
        if lables in [':DEVICE']:
            continue
        if lables in [':AREA', ':CORP', ':SUBSTATION', ':IP', ':MAC', ':REGION', ':OPENPORT', ':SCANHOLE']:
            info_list += get_short_info(device_node, node_info)
        if lables == ':DEVICECHILD':
            info_list += get_devicechild_info(device_node, node_info)
        if lables == ':PORT':
            port_list = get_port_info(device_node, node_info, ip_list)
            if len(port_list) > 0:
                info_list += port_list
                port_dict[node_info['id']] = node_info
            else:
                info_list += get_other_port_info(device_node, node_info)
        if lables == 'DEVICENAME':
            info_list += get_device_name_info(device_node, node_info)

    for node_id, node_info in node_dict.items():
        lables = str(node_info['labels'])
        if 'MESSAGE' in lables:
            info_list += get_message_info(device_node,
                                          node_info, relation_dict, port_dict)
    info_ids = {info['id']: info for info in info_list}
    new_info_list = []
    for key in key_list:
        new_info_list.append(info_ids[key])
    return new_info_list


def find_data_child(json_file, child_dir, save_dir):
    file_name = os.path.basename(json_file)
    file_name = file_name.split('.')[0]
    pkl_save_dir = os.path.join(save_dir, file_name)
    if not os.path.exists(pkl_save_dir):
        os.makedirs(pkl_save_dir)
    file_list = os.listdir(child_dir)
    data = load_json(json_file)
    data_list = []
    for item in data:
        text = item['text']
        lable = item['label']
        if 'id' in item:
            device_id = item['id']
            child_file = find_child_id(device_id, file_list)
        else:
            device_name = text.split('，')[0]
            child_file = find_child_name(device_name, file_list)
        if not child_file:
            # print('not find child file:', child_file)
            continue
        child_path = os.path.join(child_dir, child_file)
        shutil.copy(child_path, pkl_save_dir)
        with open(child_path, 'rb') as f:
            child_data = pk.load(f)
        node_data = parse_node_in_graph(child_data)
        # print(len(child_data['key_list']))
        # print(len(node_data))
        for node in node_data:
            node['label'] = lable
        child_name = os.path.basename(child_path)
        print(child_name)
        text_file = os.path.join(pkl_save_dir, child_name + '.json')
        with open(text_file, 'w', encoding='utf-8') as f:
            json.dump(node_data, f, ensure_ascii=False, indent=4)
        data_list += node_data
    save_file = os.path.join(save_dir, os.path.basename(json_file))
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    max_length = 0
    max_item = None
    min_length = 10000
    min_item = None
    types = set()
    for item in data_list:
        types.add(item['type'])
        if len(item['text']) > max_length:
            max_length = len(item['text'])
            max_item = item
            if len(item['text']) < min_length:
                min_length = len(item['text'])
                min_item = item
    print('max_length:', max_length)
    print('max_item:', max_item)
    print('min_length:', min_length)
    print('min_item:', min_item)
    print(types)


def find_child_to_text(child_dir, data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(data_dir)
    train_file = os.path.join(data_dir, 'train.json')
    val_file = os.path.join(data_dir, 'val.json')
    test_file = os.path.join(data_dir, 'test.json')
    find_data_child(train_file, child_dir, save_dir)
    find_data_child(val_file, child_dir, save_dir)
    find_data_child(test_file, child_dir, save_dir)


def all_child_to_text(child_dir, data_file, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    find_data_child(data_file, child_dir, save_dir)


if __name__ == '__main__':
    # child_dir = r'D:\SX\device_graph_3\info\child_4'
    # data_dir = r'D:\SX\classification\results\results_20240813_080040\data'
    # save_dir = data_dir + '_2'
    # find_child_to_text(child_dir, data_dir, save_dir)

    child_dir = r'D:\project\fedgraph\kg-bgcn\child' 
    data_dir = r'train_data\data'
    save_dir = r'data_node'
    
    find_child_to_text(child_dir, data_dir, save_dir)
