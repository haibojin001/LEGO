#!/usr/bin/env python3
"""
Database setup utility for creating schema and sample records.
"""

import sys
from pathlib import Path

from sqlalchemy.orm import Session

_current_backend_directory = Path(__file__).parent
if str(_current_backend_directory) not in sys.path:
    sys.path.insert(0, str(_current_backend_directory))

from ..core.config import get_data_directory, init_paths
from ..core.database import SessionLocal, get_database_url, init_database
from ..models.base import Base
from ..models.clip import Clip
from ..models.collection import Collection
from ..models.project import Project, ProjectStatus, ProjectType
from ..models.task import Task, TaskStatus, TaskType


def create_initial_data():
    """创建初始测试数据"""
    database_session = SessionLocal()

    try:
        if database_session.query(Project).count() > 0:
            print("数据库中已有数据，跳过初始数据创建")
            return

        project = Project(
            name="测试项目",
            description="这是一个测试项目，用于验证系统功能",
            project_type=ProjectType.KNOWLEDGE,
            status=ProjectStatus.PENDING,
            processing_config={
                "chunk_size": 5000,
                "min_score_threshold": 0.7,
                "max_clips_per_collection": 5,
            },
        )
        database_session.add(project)
        database_session.commit()
        database_session.refresh(project)

        database_session.add(
            Task(
                name="测试任务",
                description="测试处理任务",
                task_type=TaskType.VIDEO_PROCESSING,
                project_id=project.id,
                status=TaskStatus.PENDING,
                progress=0,
                current_step="等待开始",
                total_steps=6,
            )
        )
        database_session.add(
            Clip(
                title="测试切片",
                content="这是一个测试切片的内容",
                start_time=0,
                end_time=30,
                score=0.8,
                project_id=project.id,
            )
        )
        database_session.add(
            Collection(
                title="测试合集",
                description="这是一个测试合集",
                project_id=project.id,
            )
        )

        database_session.commit()
        print("✅ 初始测试数据创建成功")
    except Exception as error:
        print(f"❌ 创建初始数据失败: {error}")
        database_session.rollback()
    finally:
        database_session.close()


def main():
    """主函数"""
    print("🚀 开始初始化数据库...")

    init_paths()

    print(f"数据库URL: {get_database_url()}")
    print(f"数据目录: {get_data_directory()}")

    if init_database():
        print("✅ 数据库初始化成功")
        create_initial_data()
        print("🎉 数据库初始化完成！")
    else:
        print("❌ 数据库初始化失败")
        sys.exit(1)


if __name__ == "__main__":
    main()