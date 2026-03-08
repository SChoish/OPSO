# OPSO 로깅 유틸: 공통 포맷, 파일/콘솔 출력
"""
get_logger(name, log_dir=None): 이름별 로거. 포맷: [시:분:초] 메시지 (LEVEL 생략)
"""
import logging
import os
from datetime import datetime

LOG_DATEFMT = "%H:%M:%S"


def get_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(f"OPSO.{name}")
    log.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt=LOG_DATEFMT)

    # 스트림 핸들러는 없을 때만 추가 (중복 방지)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in log.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        log.addHandler(sh)

    # log_dir 이 있으면 파일 핸들러가 아직 없을 때만 추가 (파일명에 날짜+시분)
    if log_dir:
        has_file = any(isinstance(h, logging.FileHandler) for h in log.handlers)
        if not has_file:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            log_file = os.path.join(log_dir, f"training_{name}_{ts}.log")
            fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            log.addHandler(fh)
            log.info(f"로그 파일: {log_file}")

    return log
