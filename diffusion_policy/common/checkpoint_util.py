from typing import Optional, Dict
from datetime import datetime
import os
from pathlib import Path

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
                delete_value = min_value
        else:
            if value < max_value:
                delete_path = max_path
                delete_value = max_value

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                deleted_size = os.path.getsize(delete_path)
                os.remove(delete_path)
                marker_path = Path(delete_path).with_suffix('.autocleaned.txt')
                marker_path.write_text(
                    f"original_checkpoint: {delete_path}\n"
                    f"deleted_at: {datetime.now().astimezone().isoformat()}\n"
                    f"replaced_by: {ckpt_path}\n"
                    f"monitor_key: {self.monitor_key}\n"
                    f"previous_value: {delete_value}\n"
                    f"new_value: {value}\n"
                    f"reclaimed_bytes: {deleted_size}\n"
                )
            return ckpt_path
