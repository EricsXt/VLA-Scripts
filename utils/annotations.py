import os
import json
import logging
from typing import Dict, Any


class TaskVidCaptionManager:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_or_initialize()

    def _load_or_initialize(self) -> Dict[str, Any]:
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"{self.file_path} is corrupted, initializing new data")
                    return self._initialize_empty_data()
        else:
            return self._initialize_empty_data()

    def _initialize_empty_data(self) -> Dict[str, Any]:
        return {}

    def save(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def add_entry(self, task_name: str, vid: str, caption: str):
        if task_name not in self.data:
            self.data[task_name] = {}
        self.data[task_name][vid] = caption
        self.save()

    def get_caption(self, task_name: str, vid: str) -> str:
        return self.data.get(task_name, {}).get(vid)

    def get_task_vids(self, task_name: str) -> Dict[str, str]:
        return self.data.get(task_name, {})

    def remove_entry(self, task_name: str, vid: str):
        if task_name in self.data and vid in self.data[task_name]:
            del self.data[task_name][vid]
            if not self.data[task_name]:
                del self.data[task_name]
            self.save()


