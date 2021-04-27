import os
import prlqr
from datetime import datetime
from pathlib import Path
import sys
import logging.handlers

def configure_logger(settings, run_id):

    module_path = os.path.dirname(prlqr.__file__)
    log_path = module_path + '/../logs/'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_path + '/' + settings.name + '/'
    file_name = str(run_id) + '_runid_' + timestamp + '.log'

    dir = Path(path)
    dir.mkdir(parents=True, exist_ok=True)

    full_name = path + file_name

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(
        filename=full_name, maxBytes=(1048576 * 5), backupCount=7
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))