import logging
import time

import numpy as np
from ml_instrumentation.metadata import attach_metadata


def attach_metadata_with_retry(save_path, idx, meta, attempts: int = 6):
    # Every job in an experiment writes to one shared metadata db behind a lock
    # file on the network filesystem. When many jobs finish together, taking
    # the lock can fail with ESTALE ("Stale file handle"); unretried, that kills
    # the save loop and loses every not-yet-saved run in the job. The failure is
    # in lock acquisition, before any row is written, so retrying is safe.
    for attempt in range(attempts):
        try:
            attach_metadata(save_path, idx, meta)
            return
        except OSError as e:
            if attempt == attempts - 1:
                raise
            delay = 2**attempt + np.random.uniform()
            logging.getLogger("exp").warning(
                f"attach_metadata failed for idx {idx} ({e}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)
