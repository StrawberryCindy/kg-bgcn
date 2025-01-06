# -*- coding: utf-8 -*-

from py2neo import Graph, Node, NodeMatcher, RelationshipMatcher
from config import neo4j_config


class GraphSearcher:
    def __init__(self) -> None:
        self.graph = Graph(neo4j_config['url'], auth=(
            neo4j_config['user'], neo4j_config['password']), name=neo4j_config['name'])

    def search_by_class(self, class_name):
        node_list = list(NodeMatcher(self.graph).match(class_name).all())
        return node_list

    def search_relation(self, node, relation_name):
        relation = list(RelationshipMatcher(self.graph).match(
            nodes=[node], r_type=relation_name).all())
        return relation

    def search_cypher(self, cypher):
        result = self.graph.run(cypher).data()
        return result

    def get_node_labels(self):
        result = self.graph.run("CALL db.labels()")
        # 提取结果中的标签列表
        labels = [record[0] for record in result]
        # 打印所有标签
        print(labels)
        return labels

    def get_rel_labels(self):
        result = self.graph.run("CALL db.relationshipTypes()")
        # 提取结果中的标签列表
        labels = [record[0] for record in result]
        # 打印所有标签
        print(labels)
        return labels
