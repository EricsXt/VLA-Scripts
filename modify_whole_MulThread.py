import os
import sys
import json

import pandas as pd
import tensorflow_datasets as tfds

from collections import defaultdict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import tensorflow as tf
from tqdm import tqdm




def _generate_single_episode_video_threadsafe(episode_data):
    """
    线程安全的单个episode视频生成函数
    """
    try:
        episode, episode_id, instruction, save_root, fps, frame_stride, image_field, other_image_fields, dataset_nickname = episode_data
        
        import os
        import logging
        import imageio
        import numpy as np
        
        # 创建任务保存目录
        task_save_dir = os.path.join(save_root, 'samples', instruction[:30].replace(' ','_').replace('/','_'))
        os.makedirs(task_save_dir, exist_ok=True)
        
        # 生成主视角视频
        if image_field:
            try:
                from scripts.xintong.utils.video_utils import extract_frames_from_episode
                
                frames = extract_frames_from_episode(episode, image_field, frame_stride=frame_stride)
                frames_uint8 = [(f if f.dtype == np.uint8 else (f * 255).astype(np.uint8)) for f in frames]
                video_path = os.path.join(task_save_dir, f"{instruction[:30].replace(' ','_').replace('/','_')}_ep{episode_id}.mp4")
                imageio.mimsave(video_path, frames_uint8, fps=fps)
                logging.info(f"✅ 线程 {threading.current_thread().name}: 已保存主视角 ep{episode_id}")
            except Exception as e:
                logging.error(f"❌ 生成主视角视频失败 episode {episode_id}: {e}")
        
        # 生成其他视角视频
        if other_image_fields and dataset_nickname:
            # 创建其他视角保存目录
            others_save_root = os.path.join(save_root, 'others_image', instruction[:30].replace(' ','_').replace('/','_'))
            os.makedirs(others_save_root, exist_ok=True)
            
            for other_field in other_image_fields:
                try:
                    from scripts.xintong.utils.video_utils import extract_frames_from_episode
                    
                    frames = extract_frames_from_episode(episode, other_field, frame_stride=frame_stride)
                    frames_uint8 = [(f if f.dtype == np.uint8 else (f * 255).astype(np.uint8)) for f in frames]
                    video_path = os.path.join(others_save_root, f"{instruction[:30].replace(' ','_').replace('/','_')}_ep{episode_id}_{other_field}.mp4")
                    imageio.mimsave(video_path, frames_uint8, fps=fps)
                    logging.info(f"✅ 线程 {threading.current_thread().name}: 已保存其他视角 {other_field} ep{episode_id}")
                except Exception as e:
                    logging.error(f"❌ 生成其他视角 {other_field} 视频失败 episode {episode_id}: {e}")
        
        return (episode_id, instruction, True)  # 返回成功信息
        
    except Exception as e:
        logging.error(f"❌ 线程处理episode失败: {e}")
        return (episode_id, instruction, False)  # 返回失败信息


def _process_dataset_parallel(ds, save_root, fps, skip_frame, image_field, other_image_fields, dataset_nickname, manager, max_workers=4, max_queue_size=None):
    """
    并行处理数据集：主线程按顺序读取episode，多个工作线程并行生成视频
    
    Args:
        max_queue_size: 最大队列大小，控制内存占用。默认为 max_workers * 3
    """
    if max_queue_size is None:
        max_queue_size = max_workers * 3  # 默认队列大小为工作线程数的3倍
    
    task2count = defaultdict(int)
    unique_tasks = set()
    episode_id = 0
    completed_count = 0
    total_submitted = 0
    
    # 使用线程池进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="VideoGen") as executor:
        # 创建任务队列
        future_to_info = {}
        
        pbar = tqdm(desc="处理episodes并生成视频")
        
        # 创建数据集迭代器
        dataset_iter = iter(ds)
        
        try:
            while True:
                # 1. 首先处理已完成的任务，释放内存
                completed_futures = []
                for future in list(future_to_info.keys()):
                    if future.done():
                        try:
                            result_episode_id, result_instruction, success = future.result()
                            if success:
                                completed_count += 1
                            completed_futures.append(future)
                        except Exception as e:
                            logging.error(f"❌ 任务执行失败: {e}")
                            completed_futures.append(future)
                
                # 移除已完成的任务
                for future in completed_futures:
                    del future_to_info[future]
                
                # 2. 检查队列大小，如果未满则添加新任务
                if len(future_to_info) < max_queue_size:
                    try:
                        episode = next(dataset_iter)
                        
                        # 主线程：顺序读取episode并提取instruction
                        step_0 = next(iter(episode['steps']))
                        from scripts.xintong.utils.dataset_utils import extract_instruction_from_step0 as utils_extract_instruction_from_step0
                        instruction = utils_extract_instruction_from_step0(step_0).replace('\n', '')
                        if instruction == "":
                            instruction = "Dummy instruction"
                        
                        # 统计任务（主线程，保证线程安全）
                        task2count[instruction] += 1
                        unique_tasks.add(instruction)
                        
                        # 添加到manager（主线程，保证线程安全）
                        manager.add_entry(instruction, f"{episode_id}", "TODO caption")
                        
                        # 准备线程任务数据
                        episode_data = (
                            episode, episode_id, instruction, save_root, fps, skip_frame,
                            image_field, other_image_fields, dataset_nickname
                        )
                        
                        # 提交到线程池
                        future = executor.submit(_generate_single_episode_video_threadsafe, episode_data)
                        future_to_info[future] = (episode_id, instruction)
                        
                        total_submitted += 1
                        episode_id += 1
                        
                        # 更新进度条
                        pbar.set_description(f"已提交: {total_submitted}, 队列: {len(future_to_info)}, 完成: {completed_count}")
                        pbar.update(1)
                        
                    except StopIteration:
                        # 数据集遍历完成
                        logging.info(f"✅ 数据集读取完成，共读取 {episode_id} 个episodes")
                        break
                    except tf.errors.OutOfRangeError:
                        logging.info(f"✅ 数据集读取完成，共读取 {episode_id} 个episodes")
                        break
                    except Exception as e:
                        logging.error(f"❌ 主线程处理episode {episode_id} 出错: {e}")
                        episode_id += 1
                else:
                    # 队列已满，等待一小段时间再检查
                    import time
                    time.sleep(0.1)
        
        except tf.errors.OutOfRangeError:
            logging.info(f"✅ 数据集遍历完成")
        
        # 等待所有剩余任务完成
        logging.info(f"⏳ 等待剩余 {len(future_to_info)} 个视频生成任务完成...")
        for future in as_completed(future_to_info):
            try:
                result_episode_id, result_instruction, success = future.result()
                if success:
                    completed_count += 1
                pbar.set_description(f"完成 {completed_count}/{total_submitted} 个视频")
            except Exception as e:
                logging.error(f"❌ 任务执行失败: {e}")
        
        pbar.close()
    
    logging.info(f"🎉 并行处理完成: {completed_count}/{total_submitted} 个episode成功，涉及 {len(unique_tasks)} 个不同task")
    logging.info(f"📊 各task数量: {dict(task2count)}")
    
    return task2count, unique_tasks, total_submitted



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
    fps = DATASET_CONFIGS[dataset_name]['fps']
    skip_frame = DATASET_CONFIGS[dataset_name]['skip_frame'] # 跳过多少帧，默认是3帧


    # 加载数据集
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

    # 初始化annotation管理器
    annotation_file = os.path.join(save_root, 'annotations.json')
    if os.path.exists(annotation_file):
        os.remove(annotation_file)
    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2, ensure_ascii=False)  
    manager = UtilsTaskVidCaptionManager(annotation_file)

    # 并行处理数据集（默认4个工作线程）
    max_workers = 4  # 可以根据CPU核心数调整
    max_queue_size = max_workers * 2  # 控制队列大小，避免内存占用过大
    logger.info(f"🚀 开始并行处理，使用 {max_workers} 个工作线程，队列大小: {max_queue_size}")
    
    task2count, unique_tasks, total_episodes = _process_dataset_parallel(
        ds, save_root, fps, skip_frame, IMAGE,
        DATASET_CONFIGS[dataset_name].get('other_image_fields'),
        dataset, manager, max_workers, max_queue_size
    )
    
    logger.info(f"✅ {dataset_name} 处理完成: {total_episodes} 个episode，{len(unique_tasks)} 个不同task")
    manager.save()