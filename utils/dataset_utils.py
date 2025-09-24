import os
import re
import subprocess
import logging

def extract_instruction_from_step0(step_0):
    """根据字段结构提取 instruction 文本。"""
    if 'observation' not in step_0:
        return "Dummy instruction"

    obs = step_0['observation']
    if 'natural_language_instruction' in obs:
        return obs['natural_language_instruction'].numpy().decode('utf-8')
    if 'language_instruction' in step_0:
        return step_0['language_instruction'].numpy().decode('utf-8')
    instruction_bytes = obs['instruction']
    return instruction_bytes.numpy().decode('utf-8').split('\x00')[0]


def validate_tfds_dataset_path(path):
    if not os.path.exists(path):
        return False
    required_files = ['dataset_info.json', 'features.json']
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            return False
    return True


def get_local_dataset_path(dataset_name, base_path="/home2/qrchen/datasets/oxeV1.1"):
    try:
        dataset_dir = os.path.join(base_path, dataset_name)
        if not os.path.exists(dataset_dir):
            logging.error(f"本地数据集目录不存在: {dataset_dir}")
            return None

        versions = []
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                if re.match(r'^\d+\.\d+(?:\.\d+)?$', item):
                    versions.append(item)

        if not versions:
            logging.warning(f"在 {dataset_dir} 中未找到版本号目录")
            return dataset_dir

        def version_key(version_str):
            try:
                parts = version_str.split('.')
                return tuple(int(part) for part in parts)
            except ValueError:
                return (0, 0, 0)

        latest_version = max(versions, key=version_key)
        local_path = os.path.join(dataset_dir, latest_version)

        logging.info(f"本地数据集 {dataset_name}找到版本: {versions}，选择最新版本: {latest_version}")
        logging.info(f"本地路径: {local_path}")

        if not validate_tfds_dataset_path(local_path):
            logging.warning(f"路径 {local_path} 不是有效的TFDS数据集目录")
            for subdir in os.listdir(local_path):
                subdir_path = os.path.join(local_path, subdir)
                if os.path.isdir(subdir_path) and validate_tfds_dataset_path(subdir_path):
                    logging.info(f"在子目录中找到TFDS数据集: {subdir_path}")
                    return subdir_path

        return local_path
    except Exception as e:
        logging.error(f"获取本地数据集 {dataset_name} 路径时出错: {e}")
        return None


def debug_local_dataset_structure(base_path="/home2/qrchen/datasets/oxeV1.1"):
    logging.info(f"调试本地数据集结构，基础路径: {base_path}")
    if not os.path.exists(base_path):
        logging.error(f"基础路径不存在: {base_path}")
        return
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            logging.info(f"发现数据集目录: {item}")
            versions = []
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    if re.match(r'^\d+\.\d+(?:\.\d+)?$', subitem):
                        versions.append(subitem)
                        if validate_tfds_dataset_path(subitem_path):
                            logging.info(f"  ✅ 版本 {subitem}: 有效的TFDS数据集")
                        else:
                            logging.info(f"  ❌ 版本 {subitem}: 无效的TFDS数据集")
            if not versions:
                if validate_tfds_dataset_path(item_path):
                    logging.info(f"  ✅ 直接包含TFDS数据集（无版本号目录）")
                else:
                    logging.info(f"  ❌ 未发现有效的TFDS数据集结构")
            else:
                logging.info(f"  发现版本: {versions}")


def dataset2path(dataset_name, is_local=False):
    if is_local:
        return get_local_dataset_path(dataset_name)
    else:
        try:
            cmd = f'gsutil ls gs://gresearch/robotics/{dataset_name}/'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"无法访问 gs://gresearch/robotics/{dataset_name}/，错误信息: {result.stderr}")
            lines = result.stdout.strip().split('\n')
            versions = []
            for line in lines:
                if line.strip():
                    match = re.search(rf'gs://gresearch/robotics/{re.escape(dataset_name)}/(\d+\.\d+(?:\.\d+)?)', line)
                    if match:
                        versions.append(match.group(1))
            if not versions:
                raise Exception(f"无法访问 gs://gresearch/robotics/{dataset_name}/，错误信息: {result.stderr}")

            def version_key(version_str):
                try:
                    parts = version_str.split('.')
                    return tuple(int(part) for part in parts)
                except ValueError:
                    return (999, 999, 999)

            unique_versions = list(set(versions))
            max_version = max(unique_versions, key=version_key)
            logging.info(f"数据集 {dataset_name} 找到版本: {versions}，去重后: {unique_versions}，选择最大版本: {max_version}")
            logging.info(f'正常返回 gs://gresearch/robotics/{dataset_name}/{max_version}')
            return f'gs://gresearch/robotics/{dataset_name}/{max_version}'
        except Exception as e:
            logging.error(f"获取数据集 {dataset_name} 版本时出错: {e}")
            raise


