import json

from GraphSearcher import GraphSearcher


def parse_node(node):
    info = {}
    for key in node.keys():
        info[key] = node[key]
    info['id'] = node.identity
    info['labels'] = str(node.labels)
    return info


def join_node_info(node_dict, index):
    show_key = ['labels', 'id', 'name', 'NAME']
    node_id = node_dict['id']
    lable = node_dict['labels']
    result = f'{lable}节点{index}, '
    for key in show_key[2:]:
        if key in node_dict.keys():
            value = node_dict[key]
            result += f'{key}为{value}, '
    for key, value in node_dict.items():
        if key in show_key:
            continue
        if not value:
            continue
        result += f'{key}为{value}, '
    result = result[:-2] + '。'
    return result


def node_to_text(graph_search, lables):
    text_list = []
    for label in lables:
        print('Processing', label)
        node_list = graph_search.search_by_class(label)
        for i, node in enumerate(node_list):
            node_dict = parse_node(node)
            node_text = join_node_info(node_dict, i)
            node_info = {
                'text': node_text,
                'label': 0,
            }
            text_list.append(node_info)
            # if i == 3:
            #     break
            # break
        # break
    with open('sample3/node_info.json', 'w', encoding='utf-8') as f:
        json.dump(text_list, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    graph_search = GraphSearcher()
    lables = graph_search.get_node_labels()
    node_to_text(graph_search, lables)
