import os
import logging
import numpy as np
import imageio
from tqdm import tqdm


def extract_frames_from_episode(episode, image_field, max_frames=None, frame_stride=1):
    """
    从episode中提取指定图像字段的帧
    """
    steps = list(episode['steps'])
    frames = []
    for step in steps[::frame_stride]:
        if isinstance(image_field, tuple):
            field_name, sub_field_name = image_field
            img = step['observation'][field_name][sub_field_name].numpy()
        else:
            img = step['observation'][image_field].numpy()

        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            pass
        else:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
        frames.append(img)
        if (max_frames is not None) and (len(frames) >= max_frames):
            break
    return frames


def get_episode_by_id(episodes, eid):
    return episodes[eid]


def visualize_task_episodes(
    episodes,
    instruction2episodes,
    task,
    manager,
    start_idx=0,
    count=-1,
    save_root='.',
    fps=3,
    max_frames=None,
    frame_stride=4,
    image_field=None,
    other_image_fields=None,
    dataset_nickname=None,
):
    """
    可视化任务episodes，生成多个视角的视频
    """
    os.makedirs(save_root, exist_ok=True)

    # 预创建其他视角的保存目录
    if other_image_fields:
        dataset_root = os.path.dirname(os.path.dirname(save_root))
        others_save_root = os.path.join(
            dataset_root, 'others_image', task[:30].replace(' ', '_').replace('/', '_')
        )
        os.makedirs(others_save_root, exist_ok=True)

    episode_ids = instruction2episodes[task]
    selected_ids = episode_ids if count == -1 else episode_ids[start_idx:start_idx + count]
    if count != -1 and len(selected_ids) < count:
        logging.warning(f"只找到了 {len(selected_ids)} 个 episode，少于期望的 {count} 个。")

    for eid in tqdm(selected_ids, desc=f"保存 {task[:20]}"):
        manager.add_entry(task, f"{eid}", "TODO caption")
        episode = get_episode_by_id(episodes, eid)

        # 主视角
        if image_field:
            frames = extract_frames_from_episode(
                episode, image_field, max_frames=max_frames, frame_stride=frame_stride
            )
            frames_uint8 = [
                (f if f.dtype == np.uint8 else (f * 255).astype(np.uint8)) for f in frames
            ]
            video_path = os.path.join(
                save_root,
                f"{task[:30].replace(' ','_').replace('/','_')}_ep{eid}.mp4",
            )
            imageio.mimsave(video_path, frames_uint8, fps=fps)
            logging.info(f"已保存主视角: {video_path}")

        # 其他视角
        if other_image_fields and dataset_nickname:
            dataset_root = os.path.dirname(os.path.dirname(save_root))
            others_save_root = os.path.join(
                dataset_root, 'others_image', task[:30].replace(' ', '_').replace('/', '_')
            )
            os.makedirs(others_save_root, exist_ok=True)

            for other_field in other_image_fields:
                try:
                    frames = extract_frames_from_episode(
                        episode, other_field, max_frames=max_frames, frame_stride=frame_stride
                    )
                    frames_uint8 = [
                        (f if f.dtype == np.uint8 else (f * 255).astype(np.uint8)) for f in frames
                    ]
                    video_path = os.path.join(
                        others_save_root,
                        f"{task[:30].replace(' ','_').replace('/','_')}_ep{eid}_{other_field}.mp4",
                    )
                    imageio.mimsave(video_path, frames_uint8, fps=fps)
                    logging.info(f"已保存其他视角 {other_field}: {video_path}")
                except Exception as e:
                    logging.error(f"生成其他视角 {other_field} 视频时出错: {e}")


