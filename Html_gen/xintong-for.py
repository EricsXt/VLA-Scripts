import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from IPython import display
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow as tf
from tqdm import tqdm
from matplotlib import font_manager as fm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import pprint
import sys
from datetime import datetime
from jinja2 import Template

# 创建日志目录和文件
log_dir = "./xthLog"
os.makedirs(log_dir, exist_ok=True)
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"{time_stamp}.txt")

# 保存原始的stdout
original_stdout = sys.stdout

# 创建一个类来同时输出到控制台和文件
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# 打开日志文件并重定向输出
log_file = open(log_file_path, 'w', encoding='utf-8')
sys.stdout = TeeOutput(original_stdout, log_file)

print(f"日志文件已创建: {log_file_path}")
print(f"程序开始执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)


nlp = spacy.load("en_core_web_sm")
font_path = "/home2/qrchen/GillSans.ttc"
font_prop = fm.FontProperties(fname=font_path)


# dataset_name is the nickname of the dataset
def dataset2path(dataset_name):
    import subprocess
    import re
    
    try:
        # 调用 gsutil ls 命令获取所有版本
        cmd = f'gsutil ls gs://gresearch/robotics/{dataset_name}/'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"警告: 无法访问 gs://gresearch/robotics/{dataset_name}/，错误信息: {result.stderr}")
            # 如果失败，使用默认版本号逻辑
            if dataset_name == 'robo_net':
                default_version = '1.0.0'
            elif dataset_name in ['language_table', 'robo_set', 'spoc',"DROID"]:
                default_version = '0.0.1'
            elif dataset_name in ['droid']:
                default_version = '1.0.0'
            else:
                default_version = '0.1.0'
            print(f"使用默认版本号: {default_version}")
            return f'gs://gresearch/robotics/{dataset_name}/{default_version}'
        
        # 解析输出，查找版本号格式 数字.数字.数字 或 数字.数字
        lines = result.stdout.strip().split('\n')
        versions = []
        
        for line in lines:
            if line.strip():
                # 提取路径中的版本号部分
                # 格式: gs://gresearch/robotics/{dataset_name}/{version}/
                match = re.search(rf'gs://gresearch/robotics/{re.escape(dataset_name)}/(\d+\.\d+(?:\.\d+)?)', line)
                if match:
                    version_str = match.group(1)
                    versions.append(version_str)
        
        if not versions:
            print(f"警告: 未找到有效的版本号，使用默认版本")
            # 使用默认版本号逻辑
            if dataset_name == 'robo_net':
                backup_version = '1.0.0'
            elif dataset_name in ['language_table', 'robo_set', 'spoc',"DROID"]:
                backup_version = '0.0.1'
            elif dataset_name in ['droid']:
                backup_version = '1.0.0'
            else:
                backup_version = '0.1.0'
            print(f"使用默认版本号: {backup_version}")
            return f'gs://gresearch/robotics/{dataset_name}/{backup_version}'
        
        # 对版本号进行排序，找到最小版本
        def version_key(version_str):
            try:
                parts = version_str.split('.')
                return tuple(int(part) for part in parts)
            except ValueError:
                # 如果版本号格式有问题，返回一个很大的数字作为排序键
                return (999, 999, 999)
        
        # 去重版本号列表
        unique_versions = list(set(versions))
        min_version = min(unique_versions, key=version_key)
        print(f"数据集 {dataset_name} 找到版本: {versions}，去重后: {unique_versions}，选择最小版本: {min_version}")
        print(f'正常返回 gs://gresearch/robotics/{dataset_name}/{min_version}')
        return f'gs://gresearch/robotics/{dataset_name}/{min_version}'
        
    except Exception as e:
        print(f"错误: 获取数据集 {dataset_name} 版本时出错: {e}")
        # 出错时使用默认版本号逻辑
        if dataset_name == 'robo_net':
            fallback_version = '1.0.0'
        elif dataset_name in ['language_table', 'robo_set', 'spoc',"DROID"]:
            fallback_version = '0.0.1'
        elif dataset_name in ['droid']:
            fallback_version = '1.0.0'
        else:
            fallback_version = '0.1.0'
        print(f"使用默认版本号: {fallback_version}")
        print(f'异常 gs://gresearch/robotics/{dataset_name}/{min_version}')
        return f'gs://gresearch/robotics/{dataset_name}/{fallback_version}'

# ============ 工具函数 ============
def depth_to_color_img(depth):
    """将depth转为彩色图像"""
    d = depth.copy()
    d = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
    cm = plt.get_cmap('jet')
    colored = cm(d)[:, :, :3]  # 只要RGB，不要alpha
    colored = (colored * 255).astype(np.uint8)
    return colored

def as_gif(images, path="temp.gif", resize_factor=0.5):
    """生成GIF文件"""
    if resize_factor != 1.0:
        resized_images = []
        for img in images:
            width, height = img.size
            new_size = (int(width * resize_factor), int(height * resize_factor))
            resized_images.append(img.resize(new_size, Image.Resampling.LANCZOS))
        images = resized_images
    
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000/15), loop=0)
    gif_bytes = open(path,"rb").read()
    return gif_bytes

def get_language_instruction(episode, config):
    """提取语言指令"""
    for step in episode["steps"]:
        lang_inst = step[config["language_field"]].numpy()
        if isinstance(lang_inst, bytes):
            lang_inst = lang_inst.decode('utf-8')
        return lang_inst
    return ""

def center_pad_images_to_max_size(images):
    """将图像列表居中对齐并填充到最大尺寸"""
    if not images:
        return images
    
    # 找到最大的高度和宽度
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    padded_images = []
    for img in images:
        height, width = img.shape[:2]
        
        # 计算居中对齐需要的填充
        pad_height_total = max_height - height
        pad_width_total = max_width - width
        
        # 分别计算上下、左右的填充量（居中对齐）
        pad_top = pad_height_total // 2
        pad_bottom = pad_height_total - pad_top
        pad_left = pad_width_total // 2
        pad_right = pad_width_total - pad_left
        
        # 使用黑色填充 (0值)
        if len(img.shape) == 3:  # RGB图像
            padded_img = np.pad(img, 
                              ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                              mode='constant', 
                              constant_values=0)
        else:  # 灰度图像
            padded_img = np.pad(img, 
                              ((pad_top, pad_bottom), (pad_left, pad_right)), 
                              mode='constant', 
                              constant_values=0)
        
        padded_images.append(padded_img)
    
    return padded_images

def process_single_step(obs, config):
    """处理单帧数据，返回拼接的图像和深度图像"""
    # 提取RGB图像
    rgb_images = []
    for field in config["image_fields"]:
        # 处理嵌套图像路径
        if isinstance(field, tuple):
            field_name, sub_field_name = field
            rgb_images.append(obs[field_name][sub_field_name].numpy())
        else:
            rgb_images.append(obs[field].numpy())
    
    # 将RGB图像居中对齐并填充到相同尺寸后拼接
    padded_rgb_images = center_pad_images_to_max_size(rgb_images)
    concat_rgb = np.concatenate(padded_rgb_images, axis=1)
    
    # 提取并处理深度图像
    depth_images = []
    if len(config["depth_fields"]) > 0:
      for field in config["depth_fields"]:
        depth_img = obs[field].numpy()
        color_depth = depth_to_color_img(depth_img)
        depth_images.append(color_depth)
      
      # 将深度图像居中对齐并填充到相同尺寸后拼接
      padded_depth_images = center_pad_images_to_max_size(depth_images)
      concat_depth = np.concatenate(padded_depth_images, axis=1)
    else:
      concat_depth = np.zeros_like(concat_rgb)
    
    return Image.fromarray(concat_rgb), Image.fromarray(concat_depth)

def process_episode(episode, episode_idx, config):
    """处理单个episode"""
    # 获取语言指令
    lang_inst = get_language_instruction(episode, config)
    if lang_inst == "":
        return None
    
    print(f"Language Instruction: {lang_inst}")
    
    # 动态调整帧抽取
    total_frames = len(list(episode['steps']))
    frame_skip = FRAME_SKIP_LARGE if total_frames > LARGE_FRAME_THRESHOLD else FRAME_SKIP_DEFAULT
    print(f"当前episode总帧数: {total_frames}, 抽取帧数: {frame_skip}")
    
    # 收集图像
    rgb_images = []
    depth_images = []
    
    for step_idx, step in enumerate(episode["steps"]):
        if step_idx % frame_skip == 0:
            obs = step[config["observation_field"]]
            rgb_img, depth_img = process_single_step(obs, config)
            rgb_images.append(rgb_img)
            # depth_images.append(depth_img)
    
    # 生成文件名
    safe_filename = lang_inst.replace(" ", "_").replace(".", "")
    rgb_path = f"{OUTPUT_DIR}/{safe_filename}_{episode_idx}_image.gif"
    depth_path = f"{OUTPUT_DIR}/{safe_filename}_{episode_idx}_depth.gif"
    
    # 保存GIF
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    display.Image(as_gif(rgb_images, rgb_path, RESIZE_FACTOR))
    # display.Image(as_gif(depth_images, depth_path, RESIZE_FACTOR))
    print(f"已生成GIF文件: {rgb_path}, {depth_path}")
    
    return {
        "total_frames": total_frames,
        "processed_frames": len(rgb_images),
        "rgb_path": rgb_path,
        "depth_path": depth_path
    }

# 人为配置

df = pd.read_csv('/home2/qrchen/embodied-datasets/metadata.csv')

TRAIN_SPLIT = 'train[:100]'  
FRAME_SKIP_DEFAULT = 5
RESIZE_FACTOR = 1
MAX_EPISODES = 100
FRAME_SKIP_LARGE = 10  # 当帧数>100时使用
LARGE_FRAME_THRESHOLD = 100
THIRD_PERSON_FRAME_THRESHOLD = 1000

# 多数据集配置
DATASET_CONFIGS = {
    "conqhose": {
        "dataset_name_in_csv": "conqhose",
        "language_field": "language_instruction",
        "observation_field": "observation",
        "image_fields": ['frontleft_fisheye_image',"frontright_fisheye_image","hand_color_image"],
        "depth_fields": []
    },
    "FMB":{
        "dataset_name_in_csv": "FMB",
        "language_field": "language_instruction",
        "observation_field": "observation",
        "image_fields": ["image_side_1", "image_side_2", "image_wrist_1", "image_wrist_2"],
        "depth_fields": []
    },
    "Mobile-ALOHA":{
        "dataset_name_in_csv": "Mobile ALOHA",
        "language_field": "language_instruction",
        "observation_field": "observation",
        "image_fields": ['cam_high',"cam_left_wrist","cam_right_wrist"],
        "depth_fields": []
    },
    "MimicPlay":{
        "dataset_name_in_csv": "MimicPlay",
        "language_field": "language_instruction",
        "observation_field": "observation",
        "image_fields": [
            ("image", "front_image_1"),
            ("image", "front_image_2"), 
            ("wrist_image", "wrist_image")
        ],
        "depth_fields": []
    },
    "IO-AI":{
        "dataset_name_in_csv": "IO-AI",
        "language_field": "language_instruction",
        "observation_field": "observation",
        "image_fields": ["image_fisheye","image","image_left_side", "image_right_side"],
        "depth_fields": []
    },
    "RoboSet":{
        "dataset_name_in_csv": "RoboSet",
        "language_field": "language_instruction",
        "observation_field": "observation",
        "image_fields": ["image_left","image_right","image_top","image_wrist"],
        "depth_fields": []
    }
    # 可以在这里添加更多数据集配置
    # "另一个数据集名称": {
    #     "dataset_name_in_csv": "另一个数据集名称",
    #     "language_field": "对应的语言字段",
    #     "version": "1.0.0",
    #     "observation_field": "对应的观察字段",
    #     "image_fields": ['对应的图像字段'],
    #     "depth_fields": ['对应的深度字段']
    # }
}

# 遍历所有数据集配置
for dataset_key, DATASET_CONFIG in DATASET_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"开始处理数据集: {dataset_key}")
    print(f"{'='*60}")
    
    # 从df 中找到 DATASET_CONFIG 中 dataset_name_in_csv 对应的 行，得到其中 nickname 的值
    dataset_name_in_csv = DATASET_CONFIG["dataset_name_in_csv"]
    dataset = df[df['Datasets'] == dataset_name_in_csv]
    
    if dataset.empty:
        print(f"警告: 在metadata.csv中未找到数据集 {dataset_name_in_csv}，跳过此数据集")
        continue
        
    dataset = dataset['NickName'].item()
    
    OUTPUT_DIR = f"Trajectories/{dataset}/sample"

    # 构建数据
    try:
        b = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
        ds = b.as_dataset(split=TRAIN_SPLIT).shuffle(1000, seed=42)
        
        # 打印 ds 有多少条
        print(f"ds 有多少条: {len(ds)}")
        
    except Exception as e:
        print(f"错误: 无法加载数据集 {dataset}，错误信息: {e}")
        continue

    # 统计数据
    
    # 1. 统计动词和动词短语
    frame_num = []
    task_list = []
    
    # 2. 统计帧数和收集instruction
    def get_instructions_and_frame_stats(ds):
        task_list = []
        frame_num = []
        for episode_idx, episode in tqdm(enumerate(ds)):
            step_0 = list(episode['steps'])[0]
            if 'natural_language_instruction' in step_0['observation']:
                task = step_0['observation']['natural_language_instruction'].numpy().decode('utf-8')
            elif 'language_instruction' in step_0:
                task = step_0['language_instruction'].numpy().decode('utf-8')
            else:
                instruction_bytes = step_0["observation"]["instruction"]
                instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
                task = tf.strings.split(instruction_encoded, "\x00")[0].numpy().decode('utf-8')
            task_list.append(task)
            frame_num.append(len(list(episode['steps'])))
        return task_list, frame_num
    task_list, frame_num = get_instructions_and_frame_stats(ds)

    # 3. 统计帧数均值和标准差
    # 每一个episode统计帧数均值和标准差
    mean_frames = np.mean(frame_num) if frame_num else 0
    std_frames = np.std(frame_num) if frame_num else 0
    print("--------------------------------")
    print(f"Mean frames per episode: {mean_frames:.2f}")
    print(f"Standard deviation of frames: {std_frames:.2f}")
    print(f"总episode数: {len(frame_num)}，总instruction数: {len(task_list)}，去重后instruction数: {len(set(task_list))}")
    print("--------------------------------")
    print(f"所有instruction的数量:{len(task_list)}")
    print(f"所有的instruction: {task_list}")
    print(f"所有的instruction去重后数: {len(set(task_list))}")
    print(f"所有的instruction去重后:")
    for i in set(task_list):
        print(i)
    print(Counter(task_list))

    # 生产对应的gif 图片
    save_root = f"/home2/qrchen/embodied-datasets/Trajectories/{dataset}"
    os.makedirs(os.path.join(save_root, "samples"), exist_ok=True)

    frame_skip = 10

    # 每个task 最多的 trajectory 数
    MAX_TRAJECTORIES_PER_TASK = 4
    # 最多生成多少个task，如果一共有100 个task，那么我们只会输出 MAX_TASKS 个task
    MAX_TASKS = 10

    # 1. 统计每个任务的轨迹数
    file_name_list = [] # 每个task 的 trajectory 数
    task_counters = defaultdict(list) # 每个task 的 trajectory 数
    for key in Counter(task_list):
        if len(key):
            task_counters[key]

    for episode_idx, episode in tqdm(enumerate(ds)):

        # 2.1 如果任务数超过最大任务数，则跳过
        if sum(1 for lst in task_counters.values() if len(lst) > 0) > MAX_TASKS:
            break

        task = ''
        step_0 = list(episode['steps'])[0]# 获取轨迹的第一个step
        # 2.2 获取任务 遍历 嵌套字典
        if 'natural_language_instruction' in step_0['observation']: 
            task = step_0['observation']['natural_language_instruction'].numpy().decode('utf-8')
        elif 'language_instruction' in step_0:
            task = step_0['language_instruction'].numpy().decode('utf-8')
        else:
            instruction_bytes = step_0["observation"]["instruction"]
            instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
            task = tf.strings.split(instruction_encoded, "\x00")[0].numpy().decode('utf-8')

        # 2.3 如果任务为空，则跳过  
        if not len(task):
            continue

        # 2.4 如果任务数超过最大任务数，则跳过
        if task in task_counters and len(task_counters[task]) >= MAX_TRAJECTORIES_PER_TASK:
            continue

        rgb_images = []
        for step_index ,step in enumerate(episode['steps']):
            if step_index % frame_skip == 0:
                obs = step[DATASET_CONFIG["observation_field"]]
                rgb_img, depth_img = process_single_step(obs, DATASET_CONFIG) #这里掉用了 process_single_step 函数，这个函数在 debug-xintong.ipynb 中
                rgb_images.append(rgb_img)

        if task not in task_counters or len(task_counters[task]) < MAX_TRAJECTORIES_PER_TASK:
            if rgb_images:
                task_filename = task.replace(" ", "_").replace(".", "")
                
                current_count = len(task_counters[task])
                gif_path= os.path.join(save_root, "samples", task_filename, f"{current_count}.gif")
                os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                task_counters[task].append(f"{current_count}.gif")
                display.Image(as_gif(rgb_images, gif_path, RESIZE_FACTOR))

    print(f"任务计数器: {task_counters}")

    # 生成json 文件和
    with open('/home2/qrchen/embodied-datasets/Templates/script.js', 'r', encoding='utf-8') as f:
        js_template = Template(f.read())

    filled_js = js_template.render(
        gif_paths=pprint.pformat(dict(task_counters))
    )

    with open(os.path.join(save_root, 'script.js'), 'w', encoding='utf-8') as f:
        f.write(filled_js)

    # 生成 index.html 文件
    with open("/home2/qrchen/embodied-datasets/Templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    dataset_name_in_csv = DATASET_CONFIG["dataset_name_in_csv"]
    metadata = df[df['Datasets'] == dataset_name_in_csv] #得到这一行的数据

    episodes = metadata['#Trajectories'].item().replace('\n', '<br>')
    language_instructions = metadata['Language instructions'].item().replace('\n', '<br>')

    how_to_modify = metadata['How to modify it to align with our overall objectives?'].item().replace('\n', '<br>')
    contents = f"""
    <div class="info-section">
        <h3>Task Information</h3>
        <div class="info-box">
            <p>
                <span class="highlight-label">#UniqueTasks:</span> {metadata['#UniqueTasks'].item()}
                <br><br>
                <span class="highlight-label">Language Instructions:</span> {language_instructions}
            </p>
        </div>
    </div>

    <div class="info-section">
        <h3>Scene Information</h3>
        <div class="info-box">
            <p>
                <span class="highlight-label">#Scenes:</span> {metadata['#Scenes'].item()}
                <br><br>
                <span class="highlight-label">Scene Description:</span> {metadata['Scenes'].item()}
            </p>
        </div>
    </div>

    <div class="info-section">
        <h3>View Information</h3>
        <div class="info-box">
            <p>
                <span class="highlight-label"># Total Cams:</span> {int(float(metadata['# Total Cams'].item()))}
                <br>
                <span class="highlight-label"># Depth Cams:</span> {int(float(metadata['# Depth Cams'].item()))}
                <br>
                <span class="highlight-label">📄 First-person Cams:</span> {int(float(metadata['# First-person Cams'].item()))}
                <br>
                <span class="highlight-label">📄 Third-person Cams:</span> {int(float(metadata['# Third-person Cams'].item()))}
            </p>
        </div>
    </div>

    <div class="info-section">
        <h3>Dataset Size</h3>
        <div class="info-box">
            <p>
                <span class="highlight-label">#Episodes:</span> {episodes}
                <br>
                <span class="highlight-label">Avg Frames per episode:</span> {metadata['Avg. frames/trajectory'].item()}
            </p>
        </div>
    </div>

    <div class="info-section">
        <h3>How to modify?</h3>
        <div class="info-box">
            <p>
                {how_to_modify}
            </p>
        </div>
    </div>
    """

    html_content = html_content.replace("==TITLE==", metadata["Datasets"].item())
    html_content = html_content.replace("==STRUCTURE==", metadata["Full data structure"].item())
    html_content = html_content.replace("==Contents==", contents)

    # 保存到新路径
    with open(os.path.join(save_root, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"数据集 {dataset_key} 处理完成!")
    print(f"输出目录: {save_root}")

# 所有数据集处理完成
print(f"\n{'='*60}")
print("所有数据集处理完成!")
print(f"{'='*60}")

# 程序结束，清理日志重定向
print("=" * 50)
print(f"程序执行完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"所有输出已保存到: {log_file_path}")

# 恢复原始stdout并关闭日志文件
sys.stdout = original_stdout
log_file.close()

print(f"日志文件已保存: {log_file_path}")