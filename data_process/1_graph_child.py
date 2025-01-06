import os
import re
import pickle as pk
from tqdm import tqdm
from GraphSearcher import GraphSearcher


def replace_value(key, value, lables):
    if not value:
        return value
    # print(key, value)
    re_value = value
    key_1 = str(key).upper()
    pattern = r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}'
    matches = re.findall(pattern, str(value))
    if matches:
        re_value = str(value)
        for match in matches:
            re_value = re_value.replace(match, 'xxx.xxx.xxx.xxx')
    # elif 'IP' in key_1:
    #     re_value = 'xxx.xxx.xxx.xxx'
    elif 'MAC' in key_1:
        re_value = '00:00:00:00:00:00'
    elif 'GUID' in key_1 or 'DID' in key_1 or 'LID' in key_1:
        re_value = '00000000-xxx'
    if lables == ':MAC' and 'NAME' in key_1:
        re_value = '00:00:00:00:00:00'
    # print(key, re_value)
    return re_value


def parse_node(node):
    info = {}
    lables = str(node.labels)
    for key in node.keys():
        # info[key] = replace_value(key, node[key], lables)
        info[key] = node[key]

    info['id'] = node.identity
    info['labels'] = lables
    return info


def get_next_node(graph, node_id):
    node_data = {}
    relation_data = {}
    search_data = graph.search_cypher(
        f'MATCH (n)-[r]->(m) WHERE id(n)={node_id} RETURN n, m, r, type(r) as type_r')
    for result in search_data:
        node = result['n']
        relation = result['r']
        next_node = result['m']
        type_r = result['type_r']
        relation_data[relation.identity] = {
            'start': node.identity, 'end': next_node.identity, 'type': type_r}
        node_data[next_node.identity] = parse_node(next_node)
    return node_data, relation_data


def get_near_node(graph, node_id):
    node_data = {}
    relation_data = {}
    node_data_1, relation_data_1 = get_next_node(graph, node_id)
    node_data_2, relation_data_2 = get_last_node(graph, node_id)
    node_data.update(node_data_1)
    node_data.update(node_data_2)
    relation_data.update(relation_data_1)
    relation_data.update(relation_data_2)
    return node_data, relation_data


def get_last_node(graph, node_id):
    node_data = {}
    relation_data = {}
    search_data = graph.search_cypher(
        f'MATCH (n)<-[r]-(m) WHERE id(n)={node_id} RETURN n, m, r, type(r) as type_r')
    for result in search_data:
        node = result['n']
        relation = result['r']
        next_node = result['m']
        type_r = result['type_r']
        relation_data[relation.identity] = {
            'start': next_node.identity, 'end': node.identity, 'type': type_r}
        node_data[next_node.identity] = parse_node(next_node)
    return node_data, relation_data


def get_device_info(graph, port_id):
    device_name = '其他设备'
    device_lable = 'DEVICENAME'
    device_id = -1
    relation_id = -1
    cypher = f'MATCH (de:DEVICE)-[r1:has_ip]->(ip:IP)-[r2:has_port]->(port:PORT) where id(port)={
        port_id} RETURN de,r2'
    search_data = graph.search_cypher(cypher)
    for result in search_data:
        device = result['de']
        r2 = result['r2']
        device_name = device['NAME']
        device_id = device.identity
        relation_id = r2.identity
        break
    if len(search_data) == 0:
        cypher = f'MATCH (de:DEVICE)-[r1:contains]->(dec:DEVICECHILD)-[r2:has_ip]->(ip:IP)-[r3:has_port]->(port:PORT) where id(port)={
            port_id} RETURN de,r3'
        search_data = graph.search_cypher(cypher)
        for result in search_data:
            device = result['de']
            r3 = result['r3']
            device_name = device['NAME']
            device_id = device.identity
            relation_id = r3.identity
            break
    if device_id == -1 or relation_id == -1:
        cypher = f'MATCH (ip:IP)-[r1:has_port]->(port:PORT) where id(port)={
            port_id} RETURN ip,r1'
        search_data = graph.search_cypher(cypher)
        for result in search_data:
            ip = result['ip']
            r1 = result['r1']
            device_id = ip.identity
            relation_id = r1.identity
            break
    if device_id == -1 or relation_id == -1:
        return {}, {}
    info = {
        'name': device_name,
        'id': device_id,
        'labels': device_lable
    }
    node_data = {device_id: info}
    relation_data = {relation_id: {
        'start': port_id, 'end': device_id, 'type': 'belong_device'}}
    return node_data, relation_data


def get_send_message(graph, node_dict, relation_dict, ip_id):
    node_data = {}
    relation_data = {}
    cypher = f'MATCH (ip:IP)-[r1:has_port]->(port1:PORT)-[r2:send]->(m)-[r3:receive]->(port2:PORT) where id(ip)={
        ip_id} return ip,r1,port1,r2,m,r3,port2'
    search_data = graph.search_cypher(cypher)
    for result in search_data:
        r1 = result['r1']
        port1 = result['port1']
        r2 = result['r2']
        m = result['m']
        r3 = result['r3']
        port2 = result['port2']
        node_data[port1.identity] = parse_node(port1)
        node_data[port2.identity] = parse_node(port2)
        node_data[m.identity] = parse_node(m)
        relation_data[r1.identity] = {
            'start': ip_id, 'end': port1.identity, 'type': 'has_port'}
        relation_data[r2.identity] = {
            'start': port1.identity, 'end': m.identity, 'type': 'send'}
        relation_data[r3.identity] = {
            'start': m.identity, 'end': port2.identity, 'type': 'receive'
        }
        n_data, r_data = get_device_info(graph, port2.identity)
        node_data.update(n_data)
        relation_data.update(r_data)
    node_dict.update(node_data)
    relation_dict.update(relation_data)


def get_received_message(graph, node_dict, relation_dict, ip_id):
    node_data = {}
    relation_data = {}
    cypher = f'MATCH (ip:IP)-[r1:has_port]->(port1:PORT)<-[r2:receive]-(m)<-[r3:send]-(port2:PORT) where id(ip)={
        ip_id} return ip,r1,port1,r2,m,r3,port2'
    search_data = graph.search_cypher(cypher)
    for result in search_data:
        r1 = result['r1']
        port1 = result['port1']
        r2 = result['r2']
        m = result['m']
        r3 = result['r3']
        port2 = result['port2']
        node_data[port1.identity] = parse_node(port1)
        node_data[port2.identity] = parse_node(port2)
        node_data[m.identity] = parse_node(m)
        relation_data[r1.identity] = {
            'start': ip_id, 'end': port1.identity, 'type': 'has_port'}
        relation_data[r2.identity] = {
            'start': m.identity, 'end': port1.identity, 'type': 'receive'}
        relation_data[r3.identity] = {
            'start': port2.identity, 'end': m.identity, 'type': 'send'
        }
        n_data, r_data = get_device_info(graph, port2.identity)
        node_data.update(n_data)
        relation_data.update(r_data)
    node_dict.update(node_data)
    relation_dict.update(relation_data)


def get_port_path(graph, node_dict, relation_dict, node_id, direction='near'):
    if direction == 'near':
        node_data, relation_data = get_near_node(graph, node_id)
    elif direction == 'last':
        node_data, relation_data = get_last_node(graph, node_id)
    elif direction == 'next':
        node_data, relation_data = get_next_node(graph, node_id)
    else:
        return
    relation_dict.update(relation_data)
    for node_id, node in node_data.items():
        if node_id in node_dict:
            continue
        node_dict[node_id] = node
        lables = node['labels']
        if lables == ':IP':
            continue
        if lables == ':PORT':
            get_port_path(graph, node_dict, relation_dict, node_id, 'last')
        else:
            get_port_path(graph, node_dict, relation_dict, node_id)


def get_graph_child(graph, node_dict, relation_dict, device_id):
    node_data, relation_data = get_next_node(graph, device_id)
    relation_dict.update(relation_data)
    for node_id, node in node_data.items():
        if node_id in node_dict:
            continue
        lables = node['labels']
        if lables in [':DEVICETYPE', ':FACTORY']:
            continue
        node_dict[node_id] = node
        if lables in [':DEVICECHILD']:
            get_graph_child(graph, node_dict, relation_dict, node_id)
        elif lables == ':IP':
            # get_port_path(graph, node_dict, relation_dict, node_id)
            get_send_message(graph, node_dict, relation_dict, node_id)
            get_received_message(graph, node_dict, relation_dict, node_id)


def get_graph_child_list(graph, device_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # device_list = device_list[:10]
    for device in tqdm(device_list, desc='dump node'):
        device_name = device['NAME']
        if not device['FACTORY'] and not (device['DEVICECODE'] and device['DEVICECODE'] != '未知设备类型'):
            continue
        device_id = device.identity
        node_dict = {device_id: parse_node(device)}
        relation_dict = {}
        get_graph_child(graph, node_dict, relation_dict, device_id)
        # 排序后存储
        key_list = list(node_dict.keys())
        key_list.sort()
        # print(key_list)
        new_node_dict = {}
        for key in key_list:
            new_node_dict[key] = node_dict[key]
        re_key_list = list(relation_dict.keys())
        re_key_list.sort()
        new_relation_dict = {}
        for key in re_key_list:
            new_relation_dict[key] = relation_dict[key]
        conn_array = []
        for key in key_list:
            line_list = []
            for key2 in key_list:
                line_list.append(0)
            conn_array.append(line_list)
        for key, relation in new_relation_dict.items():
            start = relation['start']
            end = relation['end']
            if start not in key_list or end not in key_list:
                continue
            start_index = key_list.index(start)
            end_index = key_list.index(end)
            conn_array[start_index][end_index] = key
        dump_data = {
            'node_dict': new_node_dict,
            'relation_dict': new_relation_dict,
            'key_list': key_list,
            'conn_array': conn_array
        }
        if '/' in device_name:
            device_name = device_name.replace('/', '-')
        if '\n' in device_name:
            device_name = device_name.replace('\n', '')
        if '\\' in device_name:
            device_name = device_name.replace('\\', '')
        save_path = os.path.join(save_dir, f'{device_id}-{device_name}.pkl')
        with open(save_path, 'wb') as f:
            pk.dump(dump_data, f)
        # print(new_node_dict)
        # print(new_relation_dict)
        # break


def choose_device(device_list, corpid_list):
    new_device_list = []
    for device in device_list:
        if device['CORPID'] in corpid_list:
            new_device_list.append(device)
    return new_device_list


def fill_device_by_dump(file_dir, device_list):
    file_list = os.listdir(file_dir)
    device_id_list = [int(file.split('-')[0]) for file in file_list]
    new_device_list = []
    for device in device_list:
        device_id = device.identity
        if device_id in device_id_list:
            continue
        new_device_list.append(device)
    return new_device_list

if __name__ == '__main__':
    save_dir = r'D:\SX\device_graph_3\info\child_0822'
    # corpid_list = ['09000334', '09000351', '09000323', '09000348', '09000311',
    #                '09000325', '09000248', '09000326', '09000336']
    # corpid_list = ['09000351']
    # corpid_str = ''
    # for corpid in corpid_list:
    #     corpid_str += f'"{corpid}",'
    # print(corpid_str)
    # corpid_str = corpid_str[:-1]
    graph_search = GraphSearcher()
    # device_list = graph_search.search_cypher(f'MATCH (n:DEVICE) where n.CORPID in [{corpid_str}] RETURN n')
    device_list = graph_search.search_by_class('DEVICE')

    # device_list = choose_device(device_list, corpid_list)
    # device_list = device_list[:10]
    # print(device_list[0]['NAME'])
    dump_dir = r'D:\SX\device_graph_3\info\child_0821'
    device_list = fill_device_by_dump(dump_dir, device_list)
    print('共有设备数：', len(device_list))
    get_graph_child_list(graph_search, device_list, save_dir)
