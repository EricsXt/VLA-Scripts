import os
import json

def generate_cross_dataset_html(modifications_root, output_file='cross_dataset.html', videos_per_row=4):
    # 读取difference.json文件
    with open('/home2/qrchen/embodied-datasets/scripts/xintong/difference.json', 'r', encoding='utf-8') as f:
        all_differences = json.load(f)
    
    # 计算总的difference数量
    total_differences = len(all_differences)
    
    # 开始构建HTML
    html = [
        '<!DOCTYPE html>',
        '<html lang="zh-cn">',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>Cross Dataset Trajectory Visualization</title>',
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
        f'<h1>Cross Dataset Trajectory Visualization (Total: {total_differences} differences)</h1>',
        '<button class="top-btn" onclick="window.scrollTo({top:0,behavior:\'smooth\'});">return to top</button>'
    ]
    
    # 计数器
    difference_counter = 1
    
    # 遍历所有差异
    for difference_name, datasets in all_differences.items():
        html.append('<div class="task-block">')
        html.append('<div class="task-header">')
        html.append(f'<div class="task-title">{difference_counter}. {difference_name}</div>')
        html.append('</div>')
        
        # 创建视频展示区域
        html.append('<div class="video-pair">')
        
        # 遍历每个数据集中的视频
        for dataset_name, tasks in datasets.items():
            for task_name, episodes in tasks.items():
                for ep_id, caption in episodes.items():
                    # 构建视频路径
                    video_name = f"{task_name}_ep{ep_id}.mp4"
                    video_path = os.path.join(modifications_root, dataset_name, "samples", task_name, video_name)
                    relative_video_path = f"Modifications/{dataset_name}/samples/{task_name}/{video_name}"
                    
                    html.append('<div class="video-container">')
                    html.append('<div class="video-item">')
                    
                    # 添加视频元素
                    if os.path.exists(video_path):
                        html.append('<div class="video-wrapper">')
                        html.append(f'<video src="{relative_video_path}" controls preload="auto" width="100%" loop autoplay muted playsinline></video>')
                        html.append('</div>')
                    else:
                        html.append('<div style="background:#eee; padding:40px; text-align:center;">Video not found</div>')
                    
                    # 添加说明文字
                    html.append(f'<div class="video-caption">{caption}</div>')
                    html.append(f'<div class="dataset-label">{dataset_name}/{task_name}</div>')
                    html.append('</div>')  # .video-item
                    html.append('</div>')  # .video-container
        
        html.append('</div>')  # .video-pair
        html.append('</div>')  # .task-block
        difference_counter += 1
    
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
    output_file = "/home2/qrchen/embodied-datasets/cross_dataset.html"
    generate_cross_dataset_html(modifications_root, output_file)