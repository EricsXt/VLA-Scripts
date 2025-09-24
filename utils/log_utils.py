import os
import logging


def setup_logging(log_dir, dataset_name):
    """设置日志配置"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{dataset_name}_processing.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


