# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/8/30 14:46
Create User : 19410
Desc : xxx
"""
import os.path
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from toolkit.pre_load import neo_con


def tt01():
    db = neo_con
    entityRelation = db.getEntityRelationbyEntity("土豆")
    print(entityRelation)


if __name__ == '__main__':
    tt01()
