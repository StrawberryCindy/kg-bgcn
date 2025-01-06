# 渣渣媛的毕业设计

某硕士的毕业设计代码记录。
其中包含已发表论文 KG-BGCN 部分，源码可公开，但私有数据集涉及隐私问题，无法公开。

## 代码结构

- data_process 数据预处理模块 
    - labels 打了标签的
    - train_data  2_graph_to_text.py生成的数据
    - 其他 看123（具体处理流程见.ipynb文件）
- child 子图数据.pkl
- src 
    - bert-classification bert模型本身
    - latest_classification_model 分类最新模型
    - results 训练结果    
    main_classification.py 主训练代码