import json
import logging
import threading
from pathlib import Path

log = logging.getLogger('Config')
lock = threading.RLock()


class Config:

    # data_dir: Path = Path('H:/web-b-gone/data/')
    data_dir: Path = Path('data/')
    output_dir: Path = Path('out/')
    working_dir: Path = Path('working/')

    _save_path = Path('config.json')

    _cfg = None

    @classmethod
    def get(cls) -> 'Config':
        # print(f'get cfg {threading.current_thread().name}-{threading.current_thread().ident}')
        lock.acquire()
        cfg = cls()
        if Config._cfg is not None:
            # print(f'{threading.current_thread().name}-{threading.current_thread().ident} cfg exists')
            return Config._cfg
        if Config._save_path.exists():
            try:
                with open(Config._save_path, 'r') as f:
                    cfg_json = json.load(f)
                cfg.data_dir = Path(cfg_json.get('data_dir', cfg.data_dir))
                cfg.output_dir = Path(cfg_json.get('output_dir', cfg.output_dir))
                cfg.working_dir = Path(cfg_json.get('working_dir', cfg.working_dir))
                log.debug('Config loaded')
                # print(f'{threading.current_thread().name}-{threading.current_thread().ident} cfg loaded')
            except json.JSONDecodeError:
                pass
        else:
            log.debug('Create new config')
            # print(f'{threading.current_thread().name}-{threading.current_thread().ident} new cfg')

        cfg.save()
        Config._cfg = cfg
        lock.release()
        return cfg

    def save(self) -> None:
        log.debug('Config saved.')
        with open(Config._save_path, 'w+') as f:
            json.dump(self.to_dict(), f)

    def to_dict(self) -> dict:
        return {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'working_dir': str(self.working_dir),
        }

    def __str__(self):
        return str(self.to_dict())
