import os
import json

def generate_modifications_index(dataset_dir, videos_per_row=4):
    import os, json

    samples_dir = os.path.join(dataset_dir, 'samples')
    annotations_path = os.path.join(dataset_dir, 'annotations.json')
    if not (os.path.exists(samples_dir) and os.path.exists(annotations_path)):
        print(f"跳过 {dataset_dir}，缺少 samples 或 annotations.json")
        return

    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    html = [
        '<!DOCTYPE html>',
        '<html lang="zh-cn">',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>Modifications Dataset Viewer</title>',
        '<style>',
        "body { font-family: 'Segoe UI', Arial, sans-serif; background: #f7f7fa; padding: 30px; }",
        '.tasks-row { display: flex; flex-direction: column; gap: 48px; margin-bottom: 48px; }',
        '.task-block { background: #fff; border-radius: 14px; box-shadow: 0 2px 12px #e0e0f6; padding: 28px; width: 100%; }',
        '.instruction-origin { font-size: 1.15rem; color: #764ba2; font-weight: 600; margin-bottom: 20px; }',
        '.instruction-modified { font-size: 1rem; color: #27ae60; font-weight: 500; }',
        '.video-list { display: grid; grid-template-columns: repeat(VIDEO_COLS, 1fr); gap: 20px; margin-top: 10px; }'.replace('VIDEO_COLS', str(videos_per_row)),
        '.video-item { background: #f8faff; border-radius: 8px; box-shadow: 0 1px 6px #eee; padding: 8px; }',
        '.caption { font-size: 0.95rem; color: #666; margin-top: 4px; }',
        '.back-btn { display:inline-block; margin-bottom:18px; padding:8px 22px; background:#4e6ef2; color:#fff; border-radius:6px; text-decoration:none; font-size:1rem; font-weight:500; transition:background 0.2s; }',
        '.back-btn:hover { background:#2d4ecf; }',
        '.top-btn { position:fixed; right:36px; bottom:36px; z-index:99; background:#764ba2; color:#fff; border:none; border-radius:8px; padding:12px 22px; font-size:1.1rem; font-weight:600; cursor:pointer; box-shadow:0 2px 8px #aaa; transition:background 0.2s; }',
        '.top-btn:hover { background:#4e6ef2; }',
        '.task-buttons { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 30px; }',
        '.task-btn { padding: 10px 20px; background: #764ba2; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95rem; font-weight: 500; transition: all 0.2s; }',
        '.task-btn:hover { background: #4e6ef2; transform: translateY(-2px); }',
        '.task-btn.active { background: #27ae60; box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3); }',
        '.task-content { display: none; }',
        '.task-content.active { display: block; }',
        '.filter-btn { padding: 10px 20px; background: #e74c3c; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95rem; font-weight: 500; margin-bottom: 20px; transition: all 0.2s; }',
        '.filter-btn:hover { background: #c0392b; }',
        '.filter-btn.active { background: #2ecc71; }',
        '.todo-caption { opacity: 0.6; }',  # 为TODO标注添加特殊样式
        '</style>',
        '</head>',
        '<body>',
        '<a class="back-btn" href="../../index.html">&larr; Back</a>',
        f'<h1>{os.path.basename(dataset_dir)} Dataset Viewer</h1>',
        '<button class="filter-btn" id="filterBtn" onclick="toggleFilter()">Show Whole Dataset</button>',
        '<button class="top-btn" onclick="window.scrollTo({top:0,behavior:\'smooth\'});">返回顶部</button>'
    ]

    # 添加任务按钮区域
    html.append('<div class="task-buttons">')
    tasks = list(annotations.items())
    for i, (task, vids) in enumerate(tasks):
        if not vids:  # 跳过没有标注的任务
            continue
        slug = task[:30].replace(' ', '_').replace('/', '_')
        task_dir = os.path.join(samples_dir, slug)
        if not os.path.isdir(task_dir):
            continue
        
        # 为每个任务创建按钮
        task_id = f"task_{i}"
        html.append(f'<button class="task-btn" onclick="showTask(\'{task_id}\')">{task[:50]}{"..." if len(task) > 50 else ""}</button>')
    html.append('</div>')

    # 添加任务内容区域
    for i, (task, vids) in enumerate(tasks):
        slug = task[:30].replace(' ', '_').replace('/', '_')
        task_dir = os.path.join(samples_dir, slug)
        if not os.path.isdir(task_dir):
            continue

        # 只处理有标注的视频
        if not vids:  
            continue

        task_id = f"task_{i}"
        # 第一个任务默认显示
        display_class = "task-content active" if i == 0 else "task-content"
        
        html.append(f'<div id="{task_id}" class="{display_class}">')
        html.append('<div class="task-block">')
        html.append(f'<div class="instruction-origin">ORIGIN: {task.replace("<", "&lt;").replace(">", "&gt;")}</div>')

        video_files = [f for f in os.listdir(task_dir) if f.endswith('.mp4')]
        
        # 按 ep_id 数字大小排序视频文件
        def get_ep_id(fname):
            try:
                return int(fname.split('_ep')[-1].replace('.mp4', ''))
            except ValueError:
                return 0  # 如果解析失败，返回0
        
        video_files.sort(key=get_ep_id)
        
        html.append('<div class="video-list">')
        for fname in video_files:
            video_path = f'samples/{slug}/{fname}'
            ep_id = fname.split('_ep')[-1].replace('.mp4', '')
            caption = vids.get(ep_id, None)  # 获取标注，如果没有则为None
            
            # 只有存在标注时才显示这个视频
            if caption is not None:
                is_todo = caption.strip().upper() == "TODO CAPTION"
                todo_class = "todo-caption" if is_todo else ""
                
                html.append(f'<div class="video-item" data-todo="{str(is_todo).lower()}">')
                html.append(f'<video src="{video_path}" controls width="100%" loop autoplay muted></video>')
                html.append(f'<div class="caption"><b>Video:</b> {fname}</div>')
                html.append(f'<div class="instruction-modified {todo_class}">MODIFY : {caption}</div>')
                html.append('</div>')
        
        html.append('</div>')  # .video-list
        html.append('</div>')  # .task-block
        html.append('</div>')  # .task-content

    # 添加JavaScript代码
    html.extend([
        '<script>',
        'let showAllVideos = false;',
        '',
        'function showTask(taskId) {',
        '    // 隐藏所有任务内容',
        '    const allContents = document.querySelectorAll(".task-content");',
        '    allContents.forEach(content => content.classList.remove("active"));',
        '    ',
        '    // 移除所有按钮的active状态',
        '    const allButtons = document.querySelectorAll(".task-btn");',
        '    allButtons.forEach(btn => btn.classList.remove("active"));',
        '    ',
        '    // 显示选中的任务内容',
        '    const selectedContent = document.getElementById(taskId);',
        '    if (selectedContent) {',
        '        selectedContent.classList.add("active");',
        '    }',
        '    ',
        '    // 设置选中按钮的active状态',
        '    const selectedButton = event.target;',
        '    if (selectedButton) {',
        '        selectedButton.classList.add("active");',
        '    }',
        '}',
        '',
        'function toggleFilter() {',
        '    showAllVideos = !showAllVideos;',
        '    const filterBtn = document.getElementById("filterBtn");',
        '    ',
        '    if (showAllVideos) {',
        '        filterBtn.textContent = "Hide TODO Captions";',
        '        filterBtn.classList.add("active");',
        '        // 显示所有视频',
        '        document.querySelectorAll(".video-item[data-todo=\'true\']").forEach(item => {',
        '            item.style.display = "block";',
        '        });',
        '    } else {',
        '        filterBtn.textContent = "Show Whole Dataset";',
        '        filterBtn.classList.remove("active");',
        '        // 隐藏TODO标注的视频',
        '        document.querySelectorAll(".video-item[data-todo=\'true\']").forEach(item => {',
        '            item.style.display = "none";',
        '        });',
        '    }',
        '}',
        '',
        '// 初始隐藏TODO标注的视频',
        'document.addEventListener("DOMContentLoaded", function() {',
        '    // 初始状态：隐藏TODO标注的视频',
        '    document.querySelectorAll(".video-item[data-todo=\'true\']").forEach(item => {',
        '        item.style.display = "none";',
        '    });',
        '});',
        '</script>',
        '</body>',
        '</html>'
    ])

    with open(os.path.join(dataset_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"已生成: {os.path.join(dataset_dir, 'index.html')}")

# 用法示例
modifications_root = "/home2/qrchen/embodied-datasets/Modifications"
for dataset_name in os.listdir(modifications_root):
    dataset_dir = os.path.join(modifications_root, dataset_name)
    if not os.path.isdir(dataset_dir):
        continue
    try:
        generate_modifications_index(dataset_dir, videos_per_row=4)
    except Exception as e:
        print(f"跳过 {dataset_dir}，原因: {e}")