
def log_info(log_file, info):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(info + '\n')


def log_error(log_file, error):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(error + '\n')

