import os
import json
import base64

def generate_unified_difference_html(modifications_root, output_file='difference.html', videos_per_row=4):
    # 读取所有数据集的difference.json文件
    all_differences = {}
    # data_list = os.listdir(modifications_root)
    data_list=[
        "droid",
        "dobbe",
        "fmb",
        "aloha_mobile",
        "io_ai_tech",
        "robo_set",
        "uiuc_d3field",
        "utaustin_mutex",
        "berkeley_fanuc_manipulation",
        "cmu_playing_with_food",
        "cmu_play_fusion",
        "cmu_stretch",
]
    
    for dataset_name in data_list:
        dataset_dir = os.path.join(modifications_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
            
        difference_file = os.path.join(dataset_dir, 'difference.json')
        if not os.path.exists(difference_file):
            continue
            
        try:
            with open(difference_file, 'r', encoding='utf-8') as f:
                differences = json.load(f)
                all_differences[dataset_name] = differences
        except Exception as e:
            print(f"跳过 {dataset_name} 的difference.json，原因: {e}")
    
    # 开始构建HTML
    # 计算总的difference数量
    total_differences = sum(
        sum(1 for diff_type in task_diffs.keys() if diff_type != "description")
        for dataset_diffs in all_differences.values()
        for task_diffs in dataset_diffs.values()
    )

    html = [
        '<!DOCTYPE html>',
        '<html lang="zh-cn">',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>Unified Trajectory Differences Visualization</title>',
        '<style>',
        "body { font-family: 'Segoe UI', Arial, sans-serif; background: #f7f7fa; padding: 30px; }",
        '.tasks-row { display: flex; flex-direction: column; gap: 48px; margin-bottom: 48px; }',
        '.task-block { background: #fff; border-radius: 14px; box-shadow: 0 2px 12px #e0e0f6; padding: 28px; width: 100%; }',
        '.task-title { font-size: 1.3rem; color: #3b82f6; font-weight: 600; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #bae6fd; }',
        '.difference-section { margin-bottom: 30px; }',
        '.difference-type { font-size: 1.1rem; color: #5bb8a7; font-weight: 500; margin: 25px 0 15px 0; padding-bottom: 8px; border-bottom: 1px dashed #93e1d8; }',
        '.difference-title { font-size: 1rem; color: #3b82f6; font-weight: 500; margin: 5px 0 10px 0; }',
        '.video-pair { display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0 30px 0; }',
        '.video-container { width: calc(25% - 15px); min-width: 250px; display: flex; flex-direction: column; }',
        '.video-item { border-radius: 8px; box-shadow: 0 2px 8px rgba(96, 165, 250, 0.1); padding: 12px; background: #f0f9ff; border: 1px solid #bae6fd; }',
        '.video-caption { font-size: 1.1rem; color: #7f8c8d; margin-top: 15px; }',
        '.dataset-label { font-size: 0.9rem; color: #94a3b8; margin-top: 8px; text-align: right; font-style: italic; }',
        '.back-btn { display:inline-block; margin-bottom:18px; padding:8px 22px; background:#60a5fa; color:#fff; border-radius:6px; text-decoration:none; font-size:1rem; font-weight:500; transition:all 0.2s; }',
        '.back-btn:hover { background:#3b82f6; transform:translateY(-2px); }',
        '.top-btn { position:fixed; right:36px; bottom:36px; z-index:99; background:#5bb8a7; color:#fff; border:none; border-radius:8px; padding:12px 22px; font-size:1.1rem; font-weight:600; cursor:pointer; box-shadow:0 2px 8px rgba(91, 184, 167, 0.3); transition:all 0.2s; }',
        '.top-btn:hover { background:#4a9485; transform:translateY(-2px); }',
        '.task-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }',
        '.dataset-name { font-size: 1rem; color: #94a3b8; font-style: italic; }',
        '</style>',
        '</head>',
        '<body>',
        '<a class="back-btn" href="index.html">&larr; Back</a>',
        f'<h1>Trajectory Differences Visualization (Total: {total_differences} differences)</h1>',
        '<button class="top-btn" onclick="window.scrollTo({top:0,behavior:\'smooth\'});">return to top</button>'
    ]
    
    # 计数器已经在HTML生成时初始化了
    
    # 计数器
    difference_counter = 1
    
    # 直接显示所有内容
    for dataset_name, differences in all_differences.items():
        dataset_dir = os.path.join(modifications_root, dataset_name)
        samples_dir = os.path.join(dataset_dir, 'samples')
        
        for task, diff_types in differences.items():
            html.append('<div class="task-block">')
            html.append('<div class="task-header">')
            html.append(f'<div class="task-title">Task: {task}</div>')
            html.append(f'<div class="dataset-name">Dataset: {dataset_name}</div>')
            html.append('</div>')
            
            for diff_type, steps in diff_types.items():
                if diff_type == "description":
                    continue
                html.append('<div class="difference-section">')
                html.append(f'<div class="difference-type">{difference_counter}. Difference: {diff_type}</div>')
                difference_counter += 1
                
                if "description" in steps:
                    html.append(f'<div class="step-description">{steps["description"]}</div>')
                
                ep_pairs = [(k, v) for k, v in steps.items() if k.isdigit()]
                
                # 处理所有轨迹，每四个一组
                html.append('<div class="difference-block">')
                html.append('<div class="video-pair">')
                
                for j in range(0, len(ep_pairs)):
                    ep_id, description = ep_pairs[j]
                    slug = task[:30].replace(' ', '_').replace('/', '_')
                    task_video_dir = os.path.join(samples_dir, slug)
                    
                    video_files = []
                    if os.path.exists(task_video_dir):
                        video_files = [f for f in os.listdir(task_video_dir) 
                                     if f.endswith('.mp4') and f'_ep{ep_id}.mp4' in f]
                    
                    html.append('<div class="video-container">')
                    html.append(f'<div class="difference-title">Episode {ep_id}:</div>')
                    html.append('<div class="video-item">')
                    
                    if video_files:
                        video_path = f'Modifications/{dataset_name}/samples/{slug}/{video_files[0]}'
                        html.append('<div class="video-wrapper">')
                        html.append(f'<video src="{video_path}" controls preload="auto" width="100%" loop autoplay muted playsinline></video>')
                        html.append('</div>')
                    else:
                        html.append('<div style="background:#eee; padding:40px; text-align:center;">Video not found</div>')
                    
                    html.append(f'<div class="video-caption">{description}</div>')
                    html.append(f'<div class="dataset-label">Dataset: {dataset_name}</div>')
                    html.append('</div>')  # .video-item
                    html.append('</div>')  # .video-container
                    
                    # 每四个视频后添加新的行
                    if (j + 1) % 4 == 0 and j < len(ep_pairs) - 1:
                        html.append('</div>')  # 结束当前 video-pair
                        html.append('<div class="video-pair">')  # 开始新的一行
                
                html.append('</div>')  # .video-pair
                html.append('</div>')  # .difference-block
                html.append('</div>')  # .difference-section
            
            html.append('</div>')  # .task-block
    
    # 添加JavaScript代码
    html.extend([
        '<script>',
        'document.addEventListener("DOMContentLoaded", function() {',
        '    document.querySelectorAll("video").forEach(video => {',
        '        video.load();',
        '        const playPromise = video.play();',
        '        if (playPromise !== undefined) {',
        '            playPromise.catch(error => {',
        '                console.log("Auto-play prevented:", error);',
        '            });',
        '        }',
        '        video.addEventListener("ended", function() {',
        '            this.currentTime = 0;',
        '            this.play();',
        '        });',
        '    });',
        '});',
        '</script>',
        '</body>',
        '</html>'
    ])
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"已生成: {output_file}")

if __name__ == "__main__":
    modifications_root = "/home2/qrchen/embodied-datasets/Modifications"
    output_file = "/home2/qrchen/embodied-datasets/difference.html"
    generate_unified_difference_html(modifications_root, output_file)
