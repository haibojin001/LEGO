#!/usr/bin/env python3
import sys
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ..core.database import SessionLocal
from ..models.project import Project, ProjectStatus
from ..models.task import Task, TaskStatus
from ..services.pipeline_adapter import create_pipeline_adapter_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def execute_real_pipeline(project_id: str):
    """按照原有架构执行真实流水线"""
    logger.info(f"开始执行项目 {project_id} 的真实流水线")

    try:
        db = SessionLocal()

        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"项目 {project_id} 不存在")

            logger.info(f"验证项目存在: {project.name}")

            task = Task(
                name="真实流水线处理",
                description=f"使用原有架构处理项目 {project_id}",
                task_type="VIDEO_PROCESSING",
                project_id=project_id,
                status=TaskStatus.RUNNING,
                progress=0,
                current_step="初始化",
                total_steps=6,
            )
            db.add(task)
            db.commit()
            db.refresh(task)

            logger.info(f"任务记录已创建: {task.id}")

            project_data_path = project_root / "data" / "projects" / project_id
            video_file = project_data_path / "raw" / "input.mp4"
            subtitle_file = project_data_path / "raw" / "input.srt"

            if not video_file.exists():
                raise FileNotFoundError(f"视频文件不存在: {video_file}")
            if not subtitle_file.exists():
                raise FileNotFoundError(f"字幕文件不存在: {subtitle_file}")

            logger.info("文件路径验证成功:")
            logger.info(f"  视频: {video_file}")
            logger.info(f"  字幕: {subtitle_file}")

            adapter = create_pipeline_adapter_sync(db, str(task.id), project_id)

            logger.info("验证流水线前置条件...")
            validation_errors = adapter.validate_pipeline_prerequisites()
            if validation_errors:
                details = "; ".join(validation_errors)
                logger.error(f"流水线前置条件验证失败: {details}")
                raise ValueError(f"流水线前置条件验证失败: {details}")

            logger.info("流水线前置条件验证通过")
            logger.info("开始执行完整流水线...")

            result = adapter.process_project_sync(
                project_id=project_id,
                input_video_path=str(video_file),
                input_srt_path=str(subtitle_file),
            )

            if result.get("status") == "failed":
                failure_reason = result.get("message", "处理失败")
                logger.error(f"流水线处理失败: {failure_reason}")

                task.status = TaskStatus.FAILED
                task.error_message = failure_reason
                db.commit()

                return {
                    "success": False,
                    "error": failure_reason,
                    "result": result,
                }

            logger.info("🎉 流水线处理成功！")
            logger.info(f"处理结果: {result}")

            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.current_step = "处理完成"
            db.commit()

            return {
                "success": True,
                "result": result,
                "message": "流水线处理完成",
            }

        finally:
            db.close()

    except Exception as exc:
        error_message = f"执行流水线失败: {str(exc)}"
        logger.error(error_message)

        try:
            status_db = SessionLocal()
            latest_task = (
                status_db.query(Task)
                .filter(Task.project_id == project_id)
                .order_by(Task.created_at.desc())
                .first()
            )
            if latest_task:
                latest_task.status = TaskStatus.FAILED
                latest_task.error_message = error_message
                status_db.commit()
            status_db.close()
        except Exception as db_error:
            logger.error(f"更新任务状态失败: {db_error}")

        return {
            "success": False,
            "error": error_message,
        }


async def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python execute_real_pipeline.py <project_id>")
        sys.exit(1)

    project_id = sys.argv[1]
    result = await execute_real_pipeline(project_id)

    if result["success"]:
        print("✅ 流水线执行成功！")
        print(f"📊 结果: {result['result']}")
    else:
        print(f"❌ 流水线执行失败: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())