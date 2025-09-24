import os
import json
import base64

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
                 '.instruction-origin { font-size: 1.15rem; color: #3b82f6; font-weight: 600; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #bae6fd; }',
         '.instruction-modified { font-size: 1rem; color: #5bb8a7; font-weight: 500; }',
         '.video-list { display: grid; grid-template-columns: repeat(VIDEO_COLS, 1fr); gap: 20px; margin-top: 10px; }'.replace('VIDEO_COLS', str(videos_per_row)),
         '.video-item { background: #f0f9ff; border-radius: 8px; box-shadow: 0 2px 8px rgba(96, 165, 250, 0.1); padding: 12px; border: 1px solid #bae6fd; }',
         '.caption { font-size: 0.95rem; color: #64748b; margin-top: 8px; }',
         '.back-btn { display:inline-block; margin-bottom:18px; padding:8px 22px; background:#60a5fa; color:#fff; border-radius:6px; text-decoration:none; font-size:1rem; font-weight:500; transition:all 0.2s; }',
         '.back-btn:hover { background:#3b82f6; transform:translateY(-2px); }',
         '.top-btn { position:fixed; right:36px; bottom:36px; z-index:99; background:#5bb8a7; color:#fff; border:none; border-radius:8px; padding:12px 22px; font-size:1.1rem; font-weight:600; cursor:pointer; box-shadow:0 2px 8px rgba(91, 184, 167, 0.3); transition:all 0.2s; }',
         '.top-btn:hover { background:#4a9485; transform:translateY(-2px); }',
        '.task-buttons { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 30px; }',
        '.task-btn { padding: 10px 20px; background: #93c5fd; color: #1e3a8a; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95rem; font-weight: 500; transition: all 0.2s; }',
        '.task-btn:hover { background: #60a5fa; color: #fff; transform: translateY(-2px); }',
        '.task-btn.active { background: #5bb8a7; color: #fff; box-shadow: 0 4px 12px rgba(91, 184, 167, 0.3); }',
        '.task-content { display: none; }',
        '.task-content.active { display: block; }',
        '.filter-btn { padding: 10px 20px; background: #93c5fd; color: #1e3a8a; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95rem; font-weight: 500; margin-bottom: 20px; transition: all 0.2s; }',
        '.filter-btn:hover { background: #60a5fa; color: #fff; transform: translateY(-2px); }',
        '.filter-btn.active { background: #5bb8a7; color: #fff; box-shadow: 0 4px 12px rgba(91, 184, 167, 0.3); }',
        '.todo-caption { opacity: 0.8; color: #64748b; }',
        '.load-more-btn { display: block; margin: 20px auto; padding: 10px 20px; background: #60a5fa; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95rem; font-weight: 500; transition: all 0.2s; }',
        '.load-more-btn:hover { background: #3b82f6; transform: translateY(-2px); }',
        '.video-counter { font-size: 0.9rem; color: #7f8c8d; margin-bottom: 10px; }',
        '.hidden-video { display: none; }',
        '.pagination { display:none; }',
        '</style>',
        '</head>',
        '<body>',
        '<a class="back-btn" href="../../index.html">&larr; Back</a>',
        '<br>'
        '<a class="back-btn" href="difference.html"">&rarr;Difference Page</a>'
        f'<h1>{os.path.basename(dataset_dir)} Dataset Viewer</h1>',
        '<button class="filter-btn" id="filterBtn" onclick="toggleFilter()">Show Whole Dataset</button>',
        '<button class="top-btn" onclick="window.scrollTo({top:0,behavior:\'smooth\'});">return to top</button>'
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
    max_videos_per_task = 100  # 每个任务最多显示的视频数
    
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
                return 0
        
        video_files.sort(key=get_ep_id)
        
        # 计算实际有标注的视频数量
        valid_videos = []
        for fname in video_files:
            ep_id = fname.split('_ep')[-1].replace('.mp4', '')
            if ep_id in vids:
                valid_videos.append(fname)
        
        total_videos = len(valid_videos)
        html.append(f'<div class="video-counter">Total Videos: {total_videos} (Showing first {min(max_videos_per_task, total_videos)})</div>')
        
        html.append('<div class="video-list">')
        
        displayed_count = 0
        # 仅预渲染前100个，剩余用JS按需追加
        for fname in valid_videos[:max_videos_per_task]:
            video_path = f'samples/{slug}/{fname}'
            ep_id = fname.split('_ep')[-1].replace('.mp4', '')
            caption = vids.get(ep_id, None)
            
            if caption is not None:
                is_todo = caption.strip().upper() == "TODO CAPTION"
                todo_class = "todo-caption" if is_todo else ""
                
                html.append(f'<div class="video-item" data-todo="{str(is_todo).lower()}" data-task="{task_id}">')
                html.append(f'<video data-src="{video_path}" preload="none" controls width="100%" loop autoplay muted></video>')
                html.append(f'<div class="caption"><b>Video:</b> {fname}</div>')
                html.append(f'<div class="instruction-modified {todo_class}">MODIFY : {caption}</div>')
                html.append('</div>')
                displayed_count += 1
        
        html.append('</div>')  # .video-list
        
        # 如果视频数量超过限制，添加"Load All"按钮
        if total_videos > max_videos_per_task:
            # 将剩余视频的精简信息挂到 data-attrs 上（base64编码的JSON），供前端按批次追加
            rest_list = []
            for fname in valid_videos[max_videos_per_task:]:
                ep_id = fname.split('_ep')[-1].replace('.mp4', '')
                caption = vids.get(ep_id, None)
                if caption is None:
                    continue
                is_todo = (caption.strip().upper() == "TODO CAPTION")
                video_path = f'samples/{slug}/{fname}'
                rest_list.append({
                    'src': video_path,
                    'todo': str(is_todo).lower(),
                    'fname': fname,
                    'caption': caption,
                })
            rest_json = json.dumps(rest_list, ensure_ascii=False)
            rest_b64 = base64.b64encode(rest_json.encode('utf-8')).decode('ascii')
            html.append(f'<button class="load-more-btn" onclick="loadAllVideos(\'{task_id}\')" id="{task_id}_load" data-batch="30" data-rest-b64="{rest_b64}">Load All (show remaining {total_videos - max_videos_per_task})</button>')
        
        html.append('</div>')  # .task-block
        html.append('</div>')  # .task-content

    # 添加JavaScript代码
    html.extend([
        '<script>',
        'let showAllVideos = false;',
        '// 懒加载：用 IntersectionObserver 将 data-src 填到 video.src',
        'const io = new IntersectionObserver((entries) => {',
        '  entries.forEach(entry => {',
        '    if (entry.isIntersecting) {',
        '      const video = entry.target;',
        '      const dataSrc = video.getAttribute("data-src");',
        '      if (dataSrc && !video.src) { video.src = dataSrc; }',
        '      io.unobserve(video);',
        '    }',
        '  });',
        '}, { rootMargin: "200px" });',
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
        'function loadAllVideos(taskId) {',
        '  const btn = document.getElementById(`${taskId}_load`);',
        '  if (!btn) return;',
        '  const batch = parseInt(btn.getAttribute("data-batch")) || 30;',
        '  let rest;',
        '  try {',
        '    const b64 = btn.getAttribute("data-rest-b64") || "";',
        '    const jsonStr = b64 ? atob(b64) : "[]";',
        '    rest = JSON.parse(jsonStr);',
        '  } catch (e) { rest = []; }',
        '  if (rest.length === 0) { btn.style.display = "none"; return; }',
        '  const list = document.querySelector(`#${taskId} .video-list`);',
        '  const toAppend = rest.splice(0, batch);',
        '  // 生成并追加 DOM',
        '  toAppend.forEach(item => {',
        '    const wrapper = document.createElement("div");',
        '    wrapper.className = "video-item";',
        '    wrapper.setAttribute("data-todo", item.todo);',
        '    wrapper.setAttribute("data-task", taskId);',
        '    wrapper.innerHTML = `',
        '      <video data-src="${item.src}" preload="none" controls width="100%" loop autoplay muted></video>',
        '      <div class="caption"><b>Video:</b> ${item.fname}</div>',
        '      <div class="instruction-modified ${item.todo === "true" ? "todo-caption" : ""}">MODIFY : ${item.caption}</div>',
        '    `;',
        '    list.appendChild(wrapper);',
        '    const v = wrapper.querySelector("video");',
        '    if (v) io.observe(v);',
        '  });',
        '  // 回写剩余数据（继续以base64保存，避免引号冲突）',
        '  btn.setAttribute("data-rest-b64", btoa(JSON.stringify(rest)));',
        '  if (rest.length === 0) { btn.style.display = "none"; }',
        '}',
        '',
        '// 初始隐藏TODO标注的视频',
        'document.addEventListener("DOMContentLoaded", function() {',
        '    // 初始状态：隐藏TODO标注的视频',
        '    document.querySelectorAll(".video-item[data-todo=\'true\']").forEach(item => { item.style.display = "none"; });',
        '    // 观察现有视频做懒加载',
        '    document.querySelectorAll("video[data-src]").forEach(v => io.observe(v));',
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
    try:
        generate_modifications_index(dataset_dir, videos_per_row=4)
    except Exception as e:
        print(f"跳过 {dataset_dir}，原因: {e}")