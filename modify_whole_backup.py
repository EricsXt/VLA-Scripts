import os
import sys
import json
import logging
from collections import defaultdict

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


# 导入 utils
try:
    from scripts.xintong.utils.log_utils import setup_logging as utils_setup_logging
    from scripts.xintong.utils.dataset_utils import (
        dataset2path as utils_dataset2path,
        debug_local_dataset_structure as utils_debug_local_dataset_structure,
    )
    from scripts.xintong.utils.annotations import TaskVidCaptionManager as UtilsTaskVidCaptionManager
    from scripts.xintong.utils.video_utils import visualize_task_episodes as utils_visualize_task_episodes
    from scripts.xintong.utils.config import DATASET_CONFIGS, Local_dataset, Google_dataset
    from scripts.xintong.utils.dataset_utils import extract_instruction_from_step0 as utils_extract_instruction_from_step0
except ModuleNotFoundError:
    # 将项目根路径加入sys.path，支持直接用 `python scripts/xintong/modify_whole.py` 运行
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    from scripts.xintong.utils.log_utils import setup_logging as utils_setup_logging
    from scripts.xintong.utils.dataset_utils import (
        dataset2path as utils_dataset2path,
        debug_local_dataset_structure as utils_debug_local_dataset_structure,
    )
    from scripts.xintong.utils.annotations import TaskVidCaptionManager as UtilsTaskVidCaptionManager
    from scripts.xintong.utils.video_utils import visualize_task_episodes as utils_visualize_task_episodes
    from scripts.xintong.utils.config import DATASET_CONFIGS, Local_dataset, Google_dataset
    from scripts.xintong.utils.dataset_utils import extract_instruction_from_step0 as utils_extract_instruction_from_step0







Whether_local_dataset = True # 是否是本地数据集 false 是google数据集

df = pd.read_csv('/home2/qrchen/embodied-datasets/scripts/xintong/Datasets/metadata.csv') # 可以修改为其他路径

# 创建log文件夹
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)

#检查数据结构
if Whether_local_dataset:
    datasets_name = Local_dataset
    # 调试本地数据集结构
    print("🔍 正在检查本地数据集结构...")
    utils_debug_local_dataset_structure()
    print("=" * 60)
else:
    datasets_name = Google_dataset


# 开始逐个处理数据集
for dataset_name in datasets_name:
    dataset = dataset_name
    logger = utils_setup_logging(log_dir, dataset_name)
    logger.info(f"Processing dataset: {dataset}\n")

    # 从df 中找到 DATASET_CONFIG 中 dataset_name_in_csv 对应的行，得到其中 nickname 的值
    dataset_name_in_csv = DATASET_CONFIGS[dataset_name]['dataset_name']
    dataset = df[df['Datasets'] == dataset_name_in_csv]
    dataset = dataset['NickName'].item()
    # 设置保存路径
    save_root = os.path.join('/home2/qrchen/embodied-datasets/Modifications', dataset)
    os.makedirs(save_root, exist_ok=True)

    # 配置参数
    TRAIN_SPLIT = 'train' #基本上都是train
    MAX_EPISODES = DATASET_CONFIGS[dataset_name]['max_episodes'] # 最多收集多少个episode
    STRICT_MODE = True  # 是否严格模式
    # STRICT_MODE = True  -> 直到收齐 10 个 task 且每个 task 都 >= 10 个 episode 才停
    # STRICT_MODE = False -> 一旦遇到 10 个不同的 task 就停（不保证每个 task 都有 10 个 episode）
    # 视频相关配置
    IMAGE = DATASET_CONFIGS[dataset_name]['image_field']
    OTHER_IMAGE_FIELDS = DATASET_CONFIGS[dataset_name].get('other_image_fields', None)
    START_IDX = 0  # 从所选 task 第几个 episode 开始
    ALL_TASKS = True # 是否可视化所有的task
    fps = DATASET_CONFIGS[dataset_name]['fps']
    skip_frame = DATASET_CONFIGS[dataset_name]['skip_frame'] # 跳过多少帧，默认是3帧


    #定义三个局部变量
    task2count = defaultdict(int)
    instruction2episodes = defaultdict(list) # instruction -> [episode_id, ...]
    unique_tasks = set()  # 不同的task 

    # 加载数据集并查看其 task 和 episode 数
    dataset_path = utils_dataset2path(dataset, is_local=Whether_local_dataset)
    if dataset_path is None:
        logger.error(f"无法获取数据集路径，跳过数据集: {dataset}")
        continue
    
    if Whether_local_dataset:
        # 本地数据集加载
        logger.info(f"从本地路径加载数据集: {dataset_path}")
        b = tfds.builder_from_directory(builder_dir=dataset_path)
    else:
        # Google Cloud 数据集加载
        logger.info(f"从Google Cloud加载数据集: {dataset_path}")
        b = tfds.builder_from_directory(builder_dir=dataset_path)
    
    # 如果MAX_EPISODES大于数据集的episode数，则使用数据集的episode数
    ds = b.as_dataset(split=TRAIN_SPLIT)
    if MAX_EPISODES is None:
        max_episodes = len(ds)
    else:
        max_episodes = min(MAX_EPISODES, len(ds))
    logger.info(f"total episodes: {len(ds)}")
    ds = ds.take(max_episodes)

    episodes = []
    episode_id = 0
    pbar = tqdm(ds, desc="统计 task 构建索引并缓存 episode")

    #缓存 episode
    for episode in pbar:
        try:
            episodes.append(episode)  # 缓存 episod

            step_0 = next(iter(episode['steps']))  # 仅取第 0 步
            # 去除换行符，避免作为 task 名称时出现换行
            instruction = utils_extract_instruction_from_step0(step_0).replace('\n', '')
            # 如果是空的 ，应该补上dummy
            if instruction == "":
                instruction = "Dummy instruction"
            # 统计与索引
            task2count[instruction] += 1
            instruction2episodes[instruction].append(episode_id)
            unique_tasks.add(instruction)

        except Exception as e:
            logger.error(f"[跳过] episode {episode_id} 出错：{e}")
        finally:
            episode_id += 1

    logger.info(f"已采集到 {len(unique_tasks)} 个不同 task")
    logger.info(f"unique_tasks: {list(unique_tasks)}")

    # if STRICT_MODE:
    #     logger.info("各 task 收集 episode 数量: ")
    #     for t in list(unique_tasks):
    #         logger.info(f"  {t[:60]}... -> {len(instruction2episodes[t])}")


    # 保存 每个指令对应的 episode_id 列表
    # with open(os.path.join(save_root, "instruction2episodes.json"), "w", encoding="utf-8") as f:
    #     json.dump(instruction2episodes, f, indent=2, ensure_ascii=False)

    # 得到annotation 类

    annotation_file = os.path.join(save_root, 'annotations.json')
    # 删掉原来的annotations.json
    if os.path.exists(annotation_file):
        os.remove(annotation_file)
    # 创建新的annotations.json
    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2, ensure_ascii=False)  
    manager = UtilsTaskVidCaptionManager(annotation_file) #定义了一个类，task中 caption 标注的对应关系



    all_tasks = list(instruction2episodes.keys()) # 可视化所有的task
    start_idx = START_IDX  # 从第几个episode开始 可以查看 instruction2episodes 文件来修改
    #如果可视化all task，则可视化所有的task，否则只可视化指定的task
    if ALL_TASKS:
        for task in all_tasks:
            save_dir = os.path.join(save_root, 'samples', task[:30].replace(' ','_').replace('/','_')) #避免过长的文件名
            utils_visualize_task_episodes(
                episodes, instruction2episodes, task, 
                start_idx=start_idx, count=-1, save_root=save_dir, 
                fps=fps, frame_stride=skip_frame,
                image_field=IMAGE,
                other_image_fields=DATASET_CONFIGS[dataset_name].get('other_image_fields'),
                dataset_nickname=dataset
            )
    else:
        save_dir = os.path.join(save_root, 'samples', task[:30].replace(' ','_').replace('/','_')) #避免过长的文件名
        utils_visualize_task_episodes(
            episodes, instruction2episodes, task, 
            start_idx=start_idx, count=-1, save_root=save_dir, 
            fps=fps, frame_stride=skip_frame,   
            image_field=IMAGE,
            other_image_fields=DATASET_CONFIGS[dataset_name].get('other_image_fields'),
            dataset_nickname=dataset
        )
    manager.save()