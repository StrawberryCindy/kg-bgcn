neo4j_config = {
    "url": "http://localhost:7474",
    # "url": "http://10.0.0.154:7474",
    "user": "neo4j",
    "password": "df028823",
    # "password": "12345678",
    "name": "neo4j"
}

dataset_config = {
    'child': '../child' ,# graph_dir
    'label_data': './label_data',# train_data_dir
    'train_data': './train_data'  # save  train_dir
}

table_config = {
    'data_path': './labels',  # data_path
    'check_result': 'check_result_c2'  # lable_dir
}
