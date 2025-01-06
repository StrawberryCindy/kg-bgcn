import os
import pickle as pk
import json
import uuid
import _thread
from fastapi import Body
from datetime import datetime

# from graph_search.graph_child import get_child_thread
from config import dataset_config, table_config


def get_name(graph_data):
    node_dict = graph_data['node_dict']
    for node_id, node_info in node_dict.items():
        if node_info['labels'] == ':DEVICE':
            return node_info['name']
            # return '某设备'
    return None


def get_substation(graph_data):
    node_dict = graph_data['node_dict']
    region = ''
    substation = ''
    for node_id, node_info in node_dict.items():
        if node_info['labels'] == ':SUBSTATION':
            substation = node_info['name']
            break
    for node_id, node_info in node_dict.items():
        if node_info['labels'] == ':REGION':
            region = node_info['name']
            break
    result = region + substation
    if result:
        result = '位于' + result
    return result


def get_area(graph_data):
    node_dict = graph_data['node_dict']
    area = ''
    for node_id, node_info in node_dict.items():
        if node_info['labels'] == ':AREA':
            area = node_info['SAFE_AREA']
            break
    if area:
        area = '位于安全分区' + area
    return area


def _get_all_mac(graph_data):
    node_dict = graph_data['node_dict']
    mac_list = []
    for node_id, node_info in node_dict.items():
        if node_info['labels'] == ':MAC':
            mac_list.append(node_info['name'])
    return mac_list


def get_mac(graph_data):
    mac_list = _get_all_mac(graph_data)
    if len(mac_list) > 0:
        if len(mac_list) <= 3:
            mac = 'MAC地址为' + '、'.join(mac_list)
        else:
            mac = 'MAC地址为' + '、'.join(mac_list[:3]) + \
                '等' + str(len(mac_list)) + '个MAC地址'
    else:
        mac = ''
    return mac


def _get_guid(graph_data):
    node_dict = graph_data['node_dict']
    guid = ''
    for node_id, node_info in node_dict.items():
        if node_info['labels'] == ':DEVICE':
            guid = node_info['GUID']
            break
    return guid


def _get_all_ip(graph_data):
    node_dict = graph_data['node_dict']
    ip_list = []
    ip_id_list = []
    for node_id, node_info in node_dict.items():
        if node_info['labels'] == ':IP':
            ip_list.append(node_info['IP'])
            ip_id_list.append(node_id)
    return ip_list, ip_id_list


def get_ip(graph_data):
    ip = ''
    ip_list, _ = _get_all_ip(graph_data)
    if len(ip_list) > 0:
        if len(ip_list) <= 3:
            ip = 'IP地址为' + '、'.join(ip_list)
        else:
            ip = 'IP地址为' + '、'.join(ip_list[:3]) + \
                '等' + str(len(ip_list)) + '个IP地址'
    return ip


def get_device_child(graph_data):
    node_dict = graph_data['node_dict']
    device_child = []
    result = ''
    for node_id, node_info in node_dict.items():
        if node_info['labels'] == ':DEVICECHILD':
            device_child.append(node_info['IP'])
    if len(device_child) > 0:
        result = f'共有{len(device_child)}个子设备'
    return result


def get_openport(graph_data):
    node_dict = graph_data['node_dict']
    open_port = ''
    open_port_list = []
    for node_id, node_info in node_dict.items():
        if node_info['labels'] == ':OPENPORT':
            port_name = node_info['service']
            if port_name:
                open_port_list.append(port_name)
    if len(open_port_list) > 0:
        open_port = '开放服务有' + '、'.join(open_port_list)
    return open_port


def _get_port(graph_data, ip_id):
    node_dict = graph_data['node_dict']
    relation_dict = graph_data['relation_dict']
    port_list = []
    port_id_list = []
    for relation_id, relation_info in relation_dict.items():
        start = relation_info['start']
        end = relation_info['end']
        if not (start == ip_id):
            continue
        end_node = node_dict.get(end, {})
        if end_node:
            if end_node['labels'] == ':PORT':
                port = end_node['PORT']
                port_list.append(port)
                port_id_list.append(end)
    return port_list, port_id_list


def _get_receive_port(graph_data, m_id):
    node_dict = graph_data['node_dict']
    relation_dict = graph_data['relation_dict']
    port = ''
    device_name = ''
    port_id = ''
    for relation_id, relation_info in relation_dict.items():
        start = relation_info['start']
        end = relation_info['end']
        if not (start == m_id):
            continue
        end_node = node_dict.get(end, {})
        if end_node:
            if end_node['labels'] == ':PORT':
                port = end_node
                port_id = end
                break
    if port:
        for relation_id, relation_info in relation_dict.items():
            start = relation_info['start']
            end = relation_info['end']
            if not (start == port_id):
                continue
            end_node = node_dict.get(end, {})
            if end_node:
                if end_node['labels'] == 'DEVICENAME':
                    device_name = end_node
    return port, device_name


def _get_send_port(graph_data, m_id):
    node_dict = graph_data['node_dict']
    relation_dict = graph_data['relation_dict']
    port = ''
    device_name = ''
    port_id = ''
    for relation_id, relation_info in relation_dict.items():
        start = relation_info['start']
        end = relation_info['end']
        if not (end == m_id):
            continue
        end_node = node_dict.get(start, {})
        if end_node:
            if end_node['labels'] == ':PORT':
                port = end_node
                port_id = start
                break
    if port:
        for relation_id, relation_info in relation_dict.items():
            start = relation_info['start']
            end = relation_info['end']
            if not (start == port_id):
                continue
            end_node = node_dict.get(end, {})
            if end_node:
                if end_node['labels'] == 'DEVICENAME':
                    device_name = end_node
    return port, device_name


def _get_message(graph_data, port_id):
    node_dict = graph_data['node_dict']
    relation_dict = graph_data['relation_dict']
    send_message_list = []
    receive_message_list = []
    for relation_id, relation_info in relation_dict.items():
        start = relation_info['start']
        end = relation_info['end']
        if not (start == port_id):
            continue
        end_node = node_dict.get(end, {})
        if end_node:
            if 'MESSAGE' in end_node['labels']:
                port_node, device_name = _get_receive_port(graph_data, end)
                if port_node:
                    port = port_node['PORT']
                if device_name:
                    device_name_info = device_name['name']
                    device_id = device_name['id']
                else:
                    device_name_info = '其他设备'
                    device_id = ''
                message = end_node['applicationProto']
                info = {
                    'message': message,
                    'port': port,
                    'device': device_name_info,
                    'port_id': end,
                    'device_id': device_id,
                    'count': end_node['count'],
                    'length': int(end_node['length']),
                    'type': 'send'
                }
                send_message_list.append(info)
    for relation_id, relation_info in relation_dict.items():
        start = relation_info['start']
        end = relation_info['end']
        if not (end == port_id):
            continue
        end_node = node_dict.get(start, {})
        if end_node:
            if 'MESSAGE' in end_node['labels']:
                message = end_node['applicationProto']
                _, device_name = _get_send_port(graph_data, start)
                port_node = node_dict.get(end, {})
                if port_node:
                    port = port_node['PORT']
                if device_name:
                    device_name_info = device_name['name']
                    device_id = device_name['id']
                else:
                    device_name_info = '其他设备'
                    device_id = ''
                info = {
                    'message': message,
                    'port': port,
                    'device': device_name_info,
                    'port_id': start,
                    'device_id': device_id,
                    'count': end_node['count'],
                    'length': int(end_node['length']),
                    'type': 'receive'
                }
                receive_message_list.append(info)
    return send_message_list, receive_message_list


def _sort_message(message):
    return (message['count'], message['length'])


def _count_message(message_list):
    count_dict = {}
    for message in message_list:
        m_type = message['message']
        port = message['port']
        device_id = message['device_id']
        count_key = f'{m_type}_{port}_{device_id}'
        count = message['count']
        length = message['length']
        if count_key in count_dict:
            count_dict[count_key]['count'] += count
            count_dict[count_key]['length'] += length
        else:
            count_dict[count_key] = message.copy()
    count_list = [count_dict[key] for key in count_dict.keys()]
    return count_list


def get_device_message(graph_data):
    ip_list, ip_id_list = _get_all_ip(graph_data)
    result = []
    if len(ip_list) == 0:
        return ''
    send_message_list = []
    receive_message_list = []
    for i in range(len(ip_list)):
        ip_id = ip_id_list[i]
        port_list, port_id_list = _get_port(graph_data, ip_id)
        for i in range(len(port_list)):
            port = str(port_list[i])
            port_id = port_id_list[i]
            s_message_list, re_message_list = _get_message(
                graph_data, port_id)
            send_message_list += s_message_list
            receive_message_list += re_message_list
    count_send = _count_message(send_message_list)
    count_receive = _count_message(receive_message_list)

    message_list = count_send + count_receive
    sort_message_list = sorted(message_list, key=_sort_message, reverse=True)
    for message in sort_message_list:
        message_type = message['message']
        port = message['port']
        device_name = message['device']
        count = message['count']
        length = message['length']
        m_type = message['type']
        if port >= 32768:
            port = '随机'
        if m_type == 'send':
            if not message_type:
                result.append(f'发送{count}条消息到{device_name}的{port}端口，共{length}字节')
            else:
                result.append(f'发送{count}条{message_type}协议消息到{device_name}的{port}端口，共{length}字节')
        else:
            if not message_type:
                result.append(f'{port}端口接收到{device_name}的{count}条消息，共{length}字节')
            else:
                result.append(f'{port}端口接收到{device_name}的{count}条{message_type}协议消息，共{length}字节')
    return result


def get_label_dataset_thread(guid_list, status_key, save_dir):
    graph_dir = dataset_config['child']
    file_list = os.listdir(graph_dir)
    fsample_list = []
    for file in file_list:
        if 'VLOOKUP' in file:
            continue
        # device_id = file.split('-SPLIT-')[0]
        device_id = file.split('.pkl')[0]
        if device_id not in guid_list:
            continue
        info_list = []
        if not file.endswith('.pkl'):
            continue
        print(f'Processing {file}')
        file_path = os.path.join(graph_dir, file)
        with open(file_path, 'rb') as f:
            graph_data = pk.load(f)
        info_list.append(get_name(graph_data))
        info_list.append(get_substation(graph_data))
        info_list.append(get_area(graph_data))
        info_list.append(get_openport(graph_data))
        info_list.append(get_mac(graph_data))
        info_list.append(get_ip(graph_data))
        info_list += get_device_message(graph_data)
        result = ''
        for info in info_list:
            if info:
                result += info + '，'
            if len(result) > 512:
                break
        result = result[:-1] + '。'
        fsample = {
            'text': result,
            'id': device_id
        }
        fsample_list.append(fsample)
    count = len(fsample_list)
    # 时间戳
    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    with open(f'{save_dir}/label_{datetime_str}.json', 'w', encoding='utf-8') as f:
        json.dump(fsample_list, f, ensure_ascii=False, indent=4)
    with open(os.path.join(save_dir, 'status.json'), 'r') as f:
        status = json.load(f)
    with open(os.path.join(save_dir, 'status.json'), 'w') as f:
        status[status_key] = {'status': 'success', 'progress': count}
        json.dump(status, f)


def get_label(
    guid_list: list = Body([], description="设备的GUID列表", examples=[['guid1', 'guid2']]),
    flag: str = Body("", description="占位符", examples=[''])
):
    # 随机生成状态key
    status_key = str(uuid.uuid4())
    train_data_dir = dataset_config['label_data']
    if not os.path.exists(train_data_dir):
        os.mkdir(train_data_dir)
    save_dir = os.path.join(train_data_dir, status_key)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    status_file = os.path.join(save_dir, 'status.json')
    if not os.path.exists(status_file):
        with open(status_file, 'w') as f:
            json.dump({}, f)
    with open(status_file, 'r') as f:
        status = json.load(f)
    with open(status_file, 'w') as f:
        status[status_key] = {'status': 'running', 'progress': 0}
        json.dump(status, f)
    # 开启线程执行任务
    _thread.start_new_thread(get_label_dataset_thread, (guid_list, status_key, save_dir))
    return {'status': 'running', 'status_key': status_key}


def get_label_status(
    status_key: str = Body("", description="状态key", examples=['status_key']),
    flag: str = Body("", description="占位符", examples=[''])
):
    train_data_dir = dataset_config['label_data']
    save_dir = os.path.join(train_data_dir, status_key)
    with open(os.path.join(save_dir, 'status.json'), 'r') as f:
        status = json.load(f)
    return status.get(status_key, {'status': 'null', 'progress': 0})


def get_label_dataset(
    status_key: str = Body("", description="状态key", examples=['status_key']),
    flag: str = Body("", description="占位符", examples=[''])
):
    train_data_dir = dataset_config['label_data']
    dataset_dir = os.path.join(train_data_dir, status_key)
    file_list = os.listdir(dataset_dir)
    for file in file_list:
        if file.startswith('label_'):
            dataset_path = os.path.join(dataset_dir, file)
            dataset_path = dataset_path.replace('\\', '/')
            info = {'status': 'success', 'path': dataset_path}
            return info
    info = {'status': 'null', 'path': ''}
    return info


def get_train_dataset_thread(guid_list, all, status_key, save_dir):
    graph_dir = dataset_config['child']
    data_path = table_config['data_path']
    label_dir = table_config['check_result']
    label_path = os.path.join(data_path, label_dir + '.json')
    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    file_list = os.listdir(graph_dir)
    if all:
        guid_list = []
        for file in file_list:
            if 'VLOOKUP' in file:
                continue
            if not file.endswith('.pkl'):
                continue
            # device_id = file.split('-SPLIT-')[0]
            device_id = file.split('.pkl')[0]
            if device_id in label_data:
                guid_list.append(device_id)
    # get_child_thread(guid_list, False, status_key)
    factory_list = []
    device_type_list = []
    fsample_list = []
    tsample_list = []
    for file in file_list:
        if 'VLOOKUP' in file:
            continue
        # device_id = file.split('-SPLIT-')[0]
        device_id = file.split('.pkl')[0]
        if device_id not in guid_list:
            continue
        info_list = []
        if not file.endswith('.pkl'):
            continue
        # print(f'Processing {file}')
        file_path = os.path.join(graph_dir, file)
        with open(file_path, 'rb') as f:
            graph_data = pk.load(f)
        guid = _get_guid(graph_data)
        info_list.append(get_name(graph_data))
        info_list.append(get_substation(graph_data))
        info_list.append(get_area(graph_data))
        info_list.append(get_openport(graph_data))
        info_list.append(get_mac(graph_data))
        info_list.append(get_ip(graph_data))
        info_list += get_device_message(graph_data)
        result = ''
        for info in info_list:
            if info:
                result += info + '，'
            if len(result) > 512:
                break
        result = result[:-1] + '。'
        labels = label_data.get(guid, {})
        factory = labels.get('MARKERS_RESULTS_AS_MANUFACTURER', '')
        device_type = labels.get('MARKERS_RESULTS_AS_TYPE', '')
        # if factory and factory not in ['北京四方', '国电南瑞']:
        if factory:
            if factory in factory_list:
                flable = factory_list.index(factory)
            else:
                flable = len(factory_list)
                factory_list.append(factory)
            fsample = {
                'text': result,
                'label': flable,
                'id': device_id
            }
            fsample_list.append(fsample)
        else:
            flable = '-1'

        # if device_type and device_type not in ['远动机', '纵向加密认证装置']:
        if device_type:
            if device_type in device_type_list:
                tlabel = device_type_list.index(device_type)
            else:
                tlabel = len(device_type_list)
                device_type_list.append(device_type)
            if not tlabel:
                print(device_type)
            tsample = {
                'text': result,
                'label': tlabel,
                'id': device_id
            }
            tsample_list.append(tsample)
        else:
            tlabel = '-1'
    # print(f'{save_dir}/factory.json', fsample_list)
    with open(f'{save_dir}/factory.json', 'w', encoding='utf-8') as f:
        json.dump(fsample_list, f, ensure_ascii=False, indent=4)
    with open(f'{save_dir}/device.json', 'w', encoding='utf-8') as f:
        json.dump(tsample_list, f, ensure_ascii=False, indent=4)
    factory_dict = {i: factory for i, factory in enumerate(factory_list)}
    device_type_dict = {i: device_type for i,
                        device_type in enumerate(device_type_list)}
    with open(f'{save_dir}/factory_label.json', 'w', encoding='utf-8') as f:
        json.dump(factory_dict, f, ensure_ascii=False, indent=4)
    with open(f'{save_dir}/device_label.json', 'w', encoding='utf-8') as f:
        json.dump(device_type_dict, f, ensure_ascii=False, indent=4)
    with open(os.path.join(save_dir, 'status.json'), 'r') as f:
        status = json.load(f)
    with open(os.path.join(save_dir, 'status.json'), 'w') as f:
        status[status_key] = {'status': 'success', 'factory': len(fsample_list), 'device': len(tsample_list)}
        json.dump(status, f)


def get_train(
    guid_list: list = Body([], description="设备的GUID列表", examples=[[]]),
    all: bool = Body(False, description="是否生成全部数据集", examples=[False])
):
    # 随机生成状态key
    status_key = str(uuid.uuid4())
    train_data_dir = dataset_config['train_data']
    if not os.path.exists(train_data_dir):
        os.mkdir(train_data_dir)
    save_dir = os.path.join(train_data_dir, status_key)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    status_file = os.path.join(save_dir, 'status.json')
    if not os.path.exists(status_file):
        with open(status_file, 'w') as f:
            json.dump({}, f)
    with open(status_file, 'r') as f:
        status = json.load(f)
    with open(status_file, 'w') as f:
        status[status_key] = {'status': 'running', 'factory': 0, 'device': 0}
        json.dump(status, f)
    # 开启线程执行任务
    _thread.start_new_thread(get_train_dataset_thread, (guid_list, all, status_key, save_dir))
    return {'status': 'running', 'status_key': status_key}


def get_train_status(
    status_key: str = Body("", description="状态key", examples=['status_key']),
    flag: str = Body("", description="占位符", examples=[''])
):
    train_data_dir = dataset_config['train_data']
    save_dir = os.path.join(train_data_dir, status_key)
    with open(os.path.join(save_dir, 'status.json'), 'r') as f:
        status = json.load(f)
    return status.get(status_key, {'status': 'null', 'factory': 0, 'device': 0})


def get_train_dataset(
    status_key: str = Body("", description="状态key", examples=['status_key']),
    flag: str = Body("", description="占位符", examples=[''])
):
    train_data_dir = dataset_config['train_data']
    dataset_dir = os.path.join(train_data_dir, status_key)
    dataset_dir = dataset_dir.replace('\\', '/')
    if os.path.exists(dataset_dir):
        info = {'status': 'success', 'path': dataset_dir}
    else:
        info = {'status': 'null', 'path': ''}
    return info


if __name__ == '__main__':
    status_key = str(uuid.uuid4())
    print(status_key)
    train_data_dir = dataset_config['train_data']
    if not os.path.exists(train_data_dir):
        os.mkdir(train_data_dir)
    save_dir = os.path.join(train_data_dir, status_key)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    status_file = os.path.join(save_dir, 'status.json')
    if not os.path.exists(status_file):
        with open(status_file, 'w') as f:
            json.dump({}, f)
    with open(status_file, 'r') as f:
        status = json.load(f)
    with open(status_file, 'w') as f:
        status[status_key] = {'status': 'running', 'factory': 0, 'device': 0}
        json.dump(status, f)
    get_train_dataset_thread([], True, 1, save_dir)
