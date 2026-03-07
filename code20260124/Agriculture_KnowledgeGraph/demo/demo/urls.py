from django.urls import re_path as url
from . import index_view, index_ERform_view, detail_view
from . import tagging_data_view, tagging_data_writefile_view
from . import _404_view, overview_view
from . import relation_view
from . import tagging
from . import question_answering, decisions_making

urlpatterns = [
    url(r'^$', index_view.index),
    url(r'^ER-post', index_ERform_view.ER_post),  # 对应实体识别
    url(r'^detail', detail_view.showdetail),  # 知识概述 - 详细信息展示
    url(r'^tagging_data', tagging_data_view.showtagging_data),
    url(r'^tagging-get', tagging_data_writefile_view.tagging_push),
    url(r'^overview', overview_view.show_overview),  # 知识概述
    url(r'^404', _404_view._404_),
    url(r'^search_entity', relation_view.search_entity),  # 实体查询
    url(r'^tagging', tagging.tagging),
    url(r'^search_relation', relation_view.search_relation),  # 关系查询
    url(r'^qa', question_answering.question_answering)  # 对应的问答的后台逻辑
]
